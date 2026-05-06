import jax
import jax.numpy as jnp
from tqdm import tqdm
import functools
import optax
import equinox as eqx
import time
import jax.random as jr
from NeuralCostatePolicies.rl.utilities.normalizations import RunningStats, init_stats, update_stats, normalize_obs


def compute_gae(rewards, values, dones, next_value=0.0, next_done=False, gamma=0.99, lam=0.95):
    """Computes GAE for a single trajectory backwards through time."""
    
    def scan_fn(carry, inputs):
        gae_t_plus_1, next_val, next_d = carry
        reward_t, value_t, done_t = inputs
        
        delta_t = reward_t + gamma * next_val * (1.0 - next_d) - value_t
        gae_t = delta_t + gamma * lam * (1.0 - next_d) * gae_t_plus_1
    
        return (gae_t, value_t, done_t), gae_t

    # scan backwards through the rollout
    _, advantages = jax.lax.scan(
        scan_fn, 
        (0.0, next_value, next_done), 
        (rewards, values, dones), 
        reverse=True
    )
    returns = advantages + values
    return advantages, returns

def rollout_rnn(env, env_config, policy, key: jr.PRNGKey, init_hidden: jnp.ndarray, obs_stats: RunningStats):
    state, raw_obs, _ = env.reset(key)
    norm_obs = normalize_obs(obs_stats, raw_obs)
    
    is_first = jnp.array(True) 
        
    def step_fn(carry, _):
        # Unpack is_first instead of done
        state, norm_obs, raw_obs, current_is_first, hidden, key, running_ret = carry
        key, subkey = jr.split(key)
        
        # Pass current_is_first to the policy to trigger z_init
        action, log_prob, value, new_hidden = policy(norm_obs, current_is_first, hidden, key=subkey)
        
        clipped_action = jnp.clip(action, env.lb, env.ub)
        next_state, next_raw_obs, cost, next_done, _ = env.step(state, clipped_action, key=subkey)
        if type(next_done) is not bool:
            next_done = next_done.astype(bool)
        
        reward = -cost
        next_running_ret = reward + env_config.gamma * running_ret * (1.0 - next_done)
        next_norm_obs = normalize_obs(obs_stats, next_raw_obs)
        
        next_is_first = jnp.array(False)
        
        dictionary = {
            "state": state, "obs": norm_obs, "raw_obs": raw_obs, 
            "action": action, "reward": reward, 
            "done": next_done,             
            "is_first": current_is_first,   
            "next_obs": next_norm_obs, "value": value, 
            "log_prob": log_prob, "hidden": hidden,
            "running_ret": next_running_ret
        }
        
        return (next_state, next_norm_obs, next_raw_obs, next_is_first, new_hidden, key, next_running_ret), dictionary

    carry = (state, norm_obs, raw_obs, is_first, init_hidden, key, jnp.array(0.0))
    final_carry, data = jax.lax.scan(step_fn, carry, None, length=env.max_steps)
    
    final_obs = final_carry[1]
    final_done = jnp.array(False) 
    final_hidden = final_carry[4]
    
    return data, final_obs, final_done, final_hidden


def train_ppo_rnn(env, env_config, policy, rollout_vmap, key):
    ppo_epochs = env_config.ppo_epochs
    batch_size = env_config.batch_size
    num_trials = env_config.num_trials
    num_minibatches = env_config.num_minibatches
    learning_rate = env_config.learning_rate
    gamma = env_config.gamma 
    lam = env_config.lam 
    clip_eps = env_config.clip_eps
    vf_coef = env_config.vf_coef
    ent_coef = env_config.ent_coef
    clip_norm = env_config.clip_norm
    anneal_lr = env_config.anneal_lr
    
    # Safe fetch for costate_coef
    costate_coef = getattr(env_config, "costate_coef", 0.0)

    if anneal_lr:
        def linear_schedule(count):
            updates_per_trial = num_minibatches * ppo_epochs
            frac = 1.0 - (count // updates_per_trial) / num_trials
            return learning_rate * jnp.maximum(frac, 0.0)
        lr_schedule = linear_schedule
    else:
        lr_schedule = learning_rate
        
    optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(lr_schedule, eps=1e-5))
    params = eqx.filter(policy, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    init_hidden = jnp.zeros((batch_size, policy.hidden_layer_size))

    dummy_key = jr.PRNGKey(0)
    _, init_obs, _ = env.reset(dummy_key)
    obs_stats = init_stats(init_obs.shape)
    return_stats = init_stats((1,))

    @eqx.filter_jit
    def update_step(runner_state, obs_stats, return_stats, key, progress_fraction:float):
        policy, opt_state, current_hidden = runner_state

        target_costate_coef = getattr(env_config, "costate_coef", 0.0)
        warmup_frac = getattr(env_config, "warmup_frac", 0.0) 
        if warmup_frac > 0.0:
            warmup_multiplier = jnp.clip(progress_fraction / warmup_frac, 0.0, 1.0) # Multiplier goes from 0.0 to 1.0 linearly, then stays at 1.0
            current_costate_coef = target_costate_coef * warmup_multiplier
        else:
            current_costate_coef = target_costate_coef

        key, rollout_key, shuffle_key = jr.split(key, 3)
        keys = jr.split(rollout_key, batch_size) 
        
        data, final_obs, final_done, final_hidden = rollout_vmap(env, env_config, policy, keys, current_hidden, obs_stats)
        
        flat_raw_obs = data["raw_obs"].reshape(-1, data["raw_obs"].shape[-1])
        obs_stats = update_stats(obs_stats, flat_raw_obs)
        
        flat_returns = data["running_ret"].reshape(-1, 1)
        return_stats = update_stats(return_stats, flat_returns)
        
        obs = jax.lax.stop_gradient(data["obs"])
        actions = jax.lax.stop_gradient(data["action"])
        dones = jax.lax.stop_gradient(data["done"])
        is_firsts = jax.lax.stop_gradient(data["is_first"]) # True at t=0, False otherwise
        values = jax.lax.stop_gradient(data["value"])
        old_log_probs = jax.lax.stop_gradient(data["log_prob"])        
        
        raw_rewards = jax.lax.stop_gradient(data["reward"])
        rewards = raw_rewards / jnp.sqrt(return_stats.var[0] + 1e-8)
        
        final_values, _ = eqx.filter_vmap(policy.get_value)(final_obs, final_done, final_hidden)
        final_values = jax.lax.stop_gradient(final_values)
        
        advantages, returns = jax.vmap(functools.partial(compute_gae, gamma=gamma, lam=lam))(rewards, values, dones, final_values, final_done)
        advantages = jax.lax.stop_gradient(advantages)
        returns = jax.lax.stop_gradient(returns)

        dynamic_policy, static_policy = eqx.partition(policy, eqx.is_array)
        dynamic_opt_state, static_opt_state = eqx.partition(opt_state, eqx.is_array)

        def epoch_step(carry, epoch_key):
            dyn_pi, dyn_opt = carry
            permutation = jr.permutation(epoch_key, batch_size)
            mb_indices = permutation.reshape((num_minibatches, batch_size // num_minibatches))
            
            def update_minibatch(train_carry, mb_indices):
                d_p, d_o = train_carry
                p = eqx.combine(d_p, static_policy)
                opt = eqx.combine(d_o, static_opt_state)
                
                mb_obs = jnp.take(obs, mb_indices, axis=0)
                mb_actions = jnp.take(actions, mb_indices, axis=0)
                mb_is_firsts = jnp.take(is_firsts, mb_indices, axis=0)
                mb_dones = jnp.take(dones, mb_indices, axis=0)
                mb_adv = jnp.take(advantages, mb_indices, axis=0)
                mb_ret = jnp.take(returns, mb_indices, axis=0)
                mb_old_log_probs = jnp.take(old_log_probs, mb_indices, axis=0)
                mb_old_values = jnp.take(values, mb_indices, axis=0)
                mb_init_hidden = jnp.take(current_hidden, mb_indices, axis=0)
                
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                def loss_fn(model):
                    return_std = jnp.sqrt(return_stats.var[0] + 1e-8)
                    
                    batch_eval_seq = eqx.filter_vmap(model.evaluate_sequence, in_axes=(0,0,0,0,None))
                    new_values, new_log_probs, entropies, pmp_losses = batch_eval_seq(
                        mb_obs, mb_is_firsts, mb_actions, mb_init_hidden, return_std
                    )

                    logratio = new_log_probs - mb_old_log_probs
                    ratio = jnp.exp(logratio)

                    approx_kl = jnp.mean((ratio - 1.0) - logratio)
                    clipfrac = jnp.mean(jnp.abs(ratio - 1.0) > clip_eps)

                    loss_actor1 = ratio * mb_adv
                    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    
                    v_pred_clipped = mb_old_values + jnp.clip(new_values - mb_old_values, -clip_eps, clip_eps)
                    v_loss1 = jnp.square(new_values - mb_ret)
                    v_loss2 = jnp.square(v_pred_clipped - mb_ret)
                    loss_critic = 0.5 * jnp.maximum(v_loss1, v_loss2).mean()
                    
                    entropy_loss = entropies.mean()
                    costate_loss_mean = pmp_losses.mean()
                    
                    total_loss = (loss_actor 
                                  + vf_coef * loss_critic 
                                  - ent_coef * entropy_loss
                                  + current_costate_coef * costate_loss_mean)
                                  
                    return total_loss, (loss_actor, loss_critic, entropy_loss, approx_kl, clipfrac, costate_loss_mean)

                (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(p)
                updates, new_opt = optimizer.update(grads, opt, p)
                new_pi = eqx.apply_updates(p, updates)
                
                new_d_p, _ = eqx.partition(new_pi, eqx.is_array)
                new_d_o, _ = eqx.partition(new_opt, eqx.is_array)
                
                return (new_d_p, new_d_o), (loss, metrics)
            
            (new_dyn_pi, new_dyn_opt), (mb_losses, mb_metrics) = jax.lax.scan(
                update_minibatch, (dyn_pi, dyn_opt), mb_indices
            )
            
            epoch_loss = jnp.mean(mb_losses)
            # Averages all metrics across minibatches
            epoch_metrics = jax.tree_util.tree_map(jnp.mean, mb_metrics)
            return (new_dyn_pi, new_dyn_opt), (epoch_loss, epoch_metrics)

        epoch_keys = jr.split(shuffle_key, ppo_epochs)
        
        (final_dyn_pi, final_dyn_opt), (epoch_losses, epoch_metrics) = jax.lax.scan(
            epoch_step, (dynamic_policy, dynamic_opt_state), epoch_keys
        )
        
        final_policy = eqx.combine(final_dyn_pi, static_policy)
        final_opt_state = eqx.combine(final_dyn_opt, static_opt_state)
        
        # Take the metrics from the final epoch
        final_metrics = jax.tree_util.tree_map(lambda x: x[-1], epoch_metrics)
        
        mean_return = jnp.mean(jnp.sum(raw_rewards, axis=1)) 
        return (final_policy, final_opt_state, final_hidden), obs_stats, return_stats, (epoch_losses[-1], mean_return, final_metrics)

    print(f"Starting Recurrent PPO Training on {env.name}...")
    keys = jr.split(key, num_trials)
    pbar = tqdm(range(num_trials))
    history = {"loss": [], "return": [], "costate_loss": []}

    current_hidden = init_hidden
    for i in pbar:
        runner_state = (policy, opt_state, current_hidden)
        progress_fraction = jnp.array(i / max(1, num_trials - 1), dtype=jnp.float32)
        new_runner_state, obs_stats, return_stats, metrics = update_step(
            runner_state, obs_stats, return_stats, keys[i], progress_fraction
        )
        
        (policy, opt_state, current_hidden) = new_runner_state
        loss, mean_ret, final_metrics = metrics
        loss_actor, loss_critic, entropy_loss, approx_kl, clipfrac, costate_loss = final_metrics
        
        history["loss"].append(float(loss))
        history["return"].append(float(mean_ret))
        history["costate_loss"].append(float(costate_loss))
        
        if i % 10 == 0:
            pbar.set_postfix({"Loss": f"{round(float(loss),4)}", "Ret": f"{round(float(mean_ret), 4)}", "PMP Loss": f"{round(float(costate_loss), 4)}"})
            
    return policy, obs_stats, history

def evaluate_agent_rnn(env, policy, obs_stats, max_steps=1000):
    """
    Runs a deterministic rollout for an RNN policy.
    """
    dummy_key = jax.random.PRNGKey(0)
    state, raw_obs, _ = env.reset(dummy_key)
    
    hidden = jnp.zeros((policy.hidden_layer_size,))
    
    is_first = jnp.array(True) 
    
    state_hist, raw_obs_hist, norm_obs_hist = [], [], []
    act_hist, cost_hist, hidden_hist, done_hist = [], [], [], []    
    
    for i in range(max_steps):
        norm_obs = normalize_obs(obs_stats, raw_obs)
        
        if is_first:
            x = jax.nn.tanh(policy.encoder(norm_obs))
            current_hidden = jax.nn.tanh(policy.I_network(x))
        else:
            current_hidden = hidden
            
        raw_obs_hist.append(raw_obs)      
        norm_obs_hist.append(norm_obs)    
        hidden_hist.append(current_hidden) # Now logs z_init at t=0!       
        done_hist.append(is_first)         # Log the boundary trigger
        
        action, _, _, new_hidden = policy(norm_obs, is_first, hidden, key=None) 
        action = jnp.clip(action, env.lb, env.ub)
        
        dummy_key, step_key = jax.random.split(dummy_key)
        next_state, next_raw_obs, cost, next_done, _ = env.step(state, action, key=step_key)
        
        state_hist.append(next_state)
        act_hist.append(action)
        cost_hist.append(env.dt * cost)
        
        state = next_state
        raw_obs = next_raw_obs
        hidden = new_hidden
        
        is_first = jnp.array(False)
        
        if next_done: 
            break

    return {
        "obs": jnp.stack(raw_obs_hist),         
        "norm_obs": jnp.stack(norm_obs_hist),   
        "action": jnp.stack(act_hist),          
        "cost": jnp.stack(cost_hist),           
        "total_cost": jnp.sum(jnp.stack(cost_hist)),
        "state_hist": state_hist,
        "hidden": jnp.stack(hidden_hist),       
        "done": jnp.stack(done_hist)            
    }
