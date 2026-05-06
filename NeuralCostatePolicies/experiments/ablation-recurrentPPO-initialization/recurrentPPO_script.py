import os
import sys
import argparse
import json
import jax
import gc
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from ml_collections import config_dict

# 1. Get the absolute path to THIS script (.../gp-ode-control/gpdx/experiments/PPO)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go UP THREE levels to reach the root (.../gp-ode-control)
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

# 3. Insert the root directory at the start of Python's system path
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- Recurrent Imports ---
from mujoco_playground import registry
from NeuralCostatePolicies.rl.policy.policy import RecurrentActorCritic
from NeuralCostatePolicies.rl.environment.env import MujocoPlaygroundEnv
from NeuralCostatePolicies.rl.algorithms.PPO_recurrent import train_ppo_rnn, rollout_rnn
from NeuralCostatePolicies.rl.utilities.normalizations import init_stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1, help="The specific seed to run")
    parser.add_argument("--D_sys", type=int, required=True)
    parser.add_argument("--D_obs", type=int, required=True)
    parser.add_argument("--D_control", type=int, required=True)
    parser.add_argument("--costate_coef", type=float, required=True)
    return parser.parse_args()

def get_config(mjx_config, env_name, D_sys, D_obs, D_control, costate_coef, seed):
    config = config_dict.ConfigDict()
    config.mjx_config = mjx_config
    config.system_name = env_name
    config.D_sys = D_sys
    config.D_obs = D_obs
    config.D_control = D_control

    config.x_star = jnp.zeros((config.D_sys))
    config.u_star = jnp.zeros((config.D_control))

    config.R = jnp.eye(config.D_control) * 0.001
    config.Q = jnp.eye(config.D_sys)
    config.Q_f = jnp.eye(config.D_sys)
    config.use_mujoco_cost: bool = True
    config.cost_transform = None

    def observation(x, key):
        """ Sensor dropout with Bernoulli(p). """
        key1, key2 = jr.split(key, 2)
        bernoulli_trial = jr.bernoulli(key1, p=.5)
        return (bernoulli_trial * x) 
        
    config.observation = observation
    config.ub = 1.
    config.lb = -1.

    config.seed = seed
    config.output_dir = "./results"
    
    # dt will be set dynamically in the main loop!
    config.max_steps = 1_000         

    # Training configuration
    config.batch_size = 32
    config.hidden_layer_size = 128
    config.learning_rate = 2.5e-4
    config.learning_rate_fully_observable = 2.5e-4

    config.num_trials = 1875
    if env_name == "WalkerRun":
        config.num_trials = 3125  
    config.num_minibatches = 4     
    
    config.minibatch_size = (config.batch_size * config.max_steps) // config.num_minibatches
    config.total_timesteps = config.batch_size * config.max_steps * config.num_trials

    config.ppo_epochs = 4           
    config.gamma = 0.99 
    config.lam = 0.95 
    config.clip_eps = 0.2

    config.vf_coef = 0.5
    config.ent_coef = 0.01
    config.clip_norm = .5
    config.costate_coef = costate_coef
    config.warmup_frac = 0.
    config.anneal_lr = True

    return config

def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ablation_str = f"coef_{args.costate_coef}_warmup_0.0"
    save_dir = os.path.join(script_dir, args.task, "combined_logs_rnn", ablation_str)  
    os.makedirs(save_dir, exist_ok=True)
    
    metrics_path = os.path.join(save_dir, f"metrics_seed_{args.seed}.json")

    print(f"Running Model: Recurrent PPO | Task: {args.task} | Seed: {args.seed}")

    devices = jax.devices()
    if any(d.platform == 'gpu' for d in devices):
        print("SUCCESS: JAX is using the GPU! 🎉")
    else:
        print("WARNING: JAX is only seeing the CPU. ⚠️")

    env_name = args.task
    mjx_env = registry.load(env_name)
    mjx_env_config = registry.get_default_config(env_name)

    print(f"\n{'='*40}\n🚀 STARTING RNN SEED {args.seed}\n{'='*40}")

    key = jr.PRNGKey(args.seed)
    key, subkey = jr.split(key)

    env_config = get_config(mjx_env_config, env_name, args.D_sys, args.D_obs, args.D_control, args.costate_coef, args.seed)
    env_config.dt = mjx_env_config.ctrl_dt 
    env = MujocoPlaygroundEnv(mujoco_playground_env=mjx_env, env_config=env_config)

    # 1. Initialize Recurrent Policy
    recurrent_policy = RecurrentActorCritic(
        key=subkey, 
        in_size=env_config.D_obs, 
        hidden_size=env_config.hidden_layer_size, 
        out_size=env.D_control, 
        action_scale=env.ub
    )
    key, subkey = jr.split(key)
    
    # 2. Setup VMAP with the extra axis for hidden states
    vmap_rollout_rnn = eqx.filter_vmap(rollout_rnn, in_axes=(None, None, None, 0, 0, None))
    
    # 3. Train Recurrent PPO
    trained_rnn_ppo_policy, rnn_ppo_stats, history_rnn_ppo = train_ppo_rnn(
        env, env_config, recurrent_policy, rollout_vmap=vmap_rollout_rnn, key=subkey
    )

    model_path = os.path.join(save_dir, f"policy_seed_{args.seed}.eqx")
    stats_path = os.path.join(save_dir, f"obs_stats_seed_{args.seed}.eqx")
    
    eqx.tree_serialise_leaves(model_path, trained_rnn_ppo_policy)
    eqx.tree_serialise_leaves(stats_path, rnn_ppo_stats)
    print(f"✅ Saved RNN model and stats for seed {args.seed}")

    # Format the metrics for saving
    clean_metrics = {}
    for k, v in history_rnn_ppo.items():
        clean_metrics[k] = [float(x) for x in v]

    with open(metrics_path, "w") as f:
        json.dump(clean_metrics, f, indent=4)
        
    print(f"✅ Metrics saved for seed {args.seed} at {metrics_path}")
    print(f"\n🎉 Seed {args.seed} completed!")

if __name__ == "__main__":
    main()