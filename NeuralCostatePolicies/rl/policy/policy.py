import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
from typing import Optional, Tuple
from NeuralCostatePolicies.rl.utilities.weight_initialization import get_orthogonal_init, init_weights


class RecurrentActorCritic(eqx.Module):
    encoder: eqx.nn.Linear 
    I_network: eqx.nn.MLP #  not currently used
    gru: eqx.nn.GRUCell
    actor_out: eqx.nn.Linear  
    critic_out: eqx.nn.Linear 
    log_std: jnp.ndarray    
    hidden_layer_size: int        
    action_scale: jnp.ndarray = eqx.field(static=True)

    def __init__(self, key, in_size, hidden_size, out_size, action_scale=1.0):
        key1a, key1b, key2a, key2b, key3a, key3b, key4a, key4b, key5a, key5b = jr.split(key, 10)
        
        # Initialize networks
        self.encoder = init_weights(eqx.nn.Linear(in_size, hidden_size, key=key1a), scale=1., key=key1b)
        self.I_network = init_weights(eqx.nn.Linear(hidden_size, hidden_size, key=key5a), scale=1., key=key5b)
    
        self.gru = init_weights(eqx.nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size, use_bias=True, key=key2a), scale=1., key=key2b)
        self.actor_out = init_weights(eqx.nn.Linear(hidden_size, out_size, key=key3a), scale=.01, key=key3b)
        self.critic_out = init_weights(eqx.nn.Linear(hidden_size, 1, key=key4a), scale=1., key=key4b)

        self.log_std = jnp.zeros(out_size)
        self.hidden_layer_size = hidden_size
        
        if isinstance(action_scale, (float, int)):
            self.action_scale = jnp.ones(out_size) * float(action_scale)
        else:
            self.action_scale = action_scale

    def __call__(self, obs: jnp.ndarray, dones: jnp.ndarray, hidden: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None):
        x = jax.nn.tanh(self.encoder(obs))
        
        z_init = jax.nn.tanh(self.I_network(x))*0.
        hidden = jnp.where(dones, z_init, hidden)

        new_hidden = self.gru(x, hidden)

        mean = jax.nn.tanh(self.actor_out(new_hidden)) * self.action_scale
        value = self.critic_out(new_hidden).squeeze()
        
        if key is None:
            action = mean
            log_prob = None
        else:
            std = jnp.exp(self.log_std)
            noise = jr.normal(key, mean.shape)
            action = mean + std * noise
            
            var = std ** 2
            log_prob = -0.5 * ((action - mean) ** 2) / var - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
            log_prob = jnp.sum(log_prob)

        return action, log_prob, value, new_hidden

    def evaluate_sequence(self, obs_sequence, isfirst_sequence, action_sequence, init_hidden, return_std: float):
        """BPTT evaluation with Neural Co-state Loss Tracking."""

        def scan_step(carry_hidden, step_data):
            obs, done, action = step_data
            
            x = jax.nn.tanh(self.encoder(obs))
            
            # Boundary initialization
            z_init = jax.nn.tanh(self.I_network(x))*0.
            current_hidden = jnp.where(done, z_init, carry_hidden)

            #  NCO loss ( co-state target)
            def get_value_fn(obs_feat, h_state):
                h_next = self.gru(obs_feat, h_state)
                return self.critic_out(h_next).squeeze()
            
            value = get_value_fn(x, current_hidden)
            
            lambda_target_scaled = jax.lax.stop_gradient(jax.grad(get_value_fn, argnums=0)(x, current_hidden))
            lambda_target = lambda_target_scaled * return_std

            eps = 1e-8
            dot_product = jnp.sum(current_hidden * lambda_target)
            norm_z = jnp.sqrt(jnp.sum(jnp.square(current_hidden)) + eps)
            norm_lambda = jnp.sqrt(jnp.sum(jnp.square(lambda_target)) + eps)
            
            cos_sim = dot_product / (norm_z * norm_lambda)
            pmp_loss = 1.0 - cos_sim

            # Actor pass
            new_hidden = self.gru(x, current_hidden)
            mean = jax.nn.tanh(self.actor_out(new_hidden)) * self.action_scale
        
            std = jnp.exp(self.log_std)
            var = std ** 2
            log_prob = -0.5 * ((action - mean) ** 2) / var - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
            log_prob = jnp.sum(log_prob) 
            entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi * var))
            
            return new_hidden, (value, log_prob, entropy, pmp_loss)

        _, (values, log_probs, entropies, pmp_losses) = jax.lax.scan(
            scan_step, init_hidden, (obs_sequence, isfirst_sequence, action_sequence)
        )
        
        return values, log_probs, entropies, pmp_losses

    def get_value(self, obs, done, hidden):
        x = jax.nn.tanh(self.encoder(obs))
        z_init = jax.nn.tanh(self.I_network(x))
        hidden = jnp.where(done, z_init, hidden)
        new_hidden = self.gru(x, hidden)
        value = self.critic_out(new_hidden).squeeze()
        return value, new_hidden    


class CTRNNCell(eqx.Module):
    """ Continuous-Time RNN Cell """
    W_in: eqx.nn.Linear
    W_rec: eqx.nn.Linear
    log_alpha: jnp.ndarray 

    def __init__(self, in_size, hidden_size, key):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.W_in = init_weights(eqx.nn.Linear(in_size, hidden_size, use_bias=True, key=k1), scale=1., key=k2)
        self.W_rec = init_weights(eqx.nn.Linear(hidden_size, hidden_size, use_bias=False, key=k3), scale=1., key=k4)
        self.log_alpha = jnp.zeros(hidden_size)   # Initialize log_alpha to 0 (sigmoid(0) = 0.5 default leak rate)

    def __call__(self, x, hidden):
        # Bound alpha between 0 and 1 to ensure stable leaky integration
        alpha = jax.nn.sigmoid(self.log_alpha)
        target = jax.nn.tanh(self.W_in(x) + self.W_rec(hidden))
        new_hidden = (1.0 - alpha) * hidden + alpha * target
        
        return new_hidden


class CTRNNActorCritic(eqx.Module):
    encoder: eqx.nn.Linear 
    I_network: eqx.nn.Linear # not currently used.
    ctrnn: CTRNNCell          
    actor_out: eqx.nn.Linear  
    critic_out: eqx.nn.Linear 
    log_std: jnp.ndarray           
    hidden_layer_size: int 
    action_scale: jnp.ndarray = eqx.field(static=True)

    def __init__(self, key, in_size, hidden_size, out_size, action_scale=1.0):
        key1a, key1b, key2, key3a, key3b, key4a, key4b, key5a, key5b = jr.split(key, 9)
        
        # Initialize networks
        self.encoder = init_weights(eqx.nn.Linear(in_size, hidden_size, key=key1a), scale=1., key=key1b)
        self.I_network = init_weights(eqx.nn.Linear(hidden_size, hidden_size, key=key5a), scale=1., key=key5b)
        
        self.ctrnn = CTRNNCell(in_size=hidden_size, hidden_size=hidden_size, key=key2)
        self.actor_out = init_weights(eqx.nn.Linear(hidden_size, out_size, key=key3a), scale=.01, key=key3b)
        self.critic_out = init_weights(eqx.nn.Linear(hidden_size, 1, key=key4a), scale=1., key=key4b)

        self.log_std = jnp.zeros(out_size) 
        self.hidden_layer_size = hidden_size
        
        if isinstance(action_scale, (float, int)):
            self.action_scale = jnp.ones(out_size) * float(action_scale)
        else:
            self.action_scale = action_scale

    def __call__(self, obs: jnp.ndarray, dones: jnp.ndarray, hidden: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None):
        """Returns action and value."""
        x = jax.nn.tanh(self.encoder(obs))
        
        z_init = jax.nn.tanh(self.I_network(x)) *0.
        hidden = jnp.where(dones, z_init, hidden)

        new_hidden = self.ctrnn(x, hidden)

        mean = jax.nn.tanh(self.actor_out(new_hidden)) * self.action_scale
        value = self.critic_out(new_hidden).squeeze()
        
        if key is None:
            action = mean
            log_prob = None
        else:
            std = jnp.exp(self.log_std)
            noise = jr.normal(key, mean.shape)
            action = mean + std * noise
            
            var = std ** 2
            log_prob = -0.5 * ((action - mean) ** 2) / var - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
            log_prob = jnp.sum(log_prob)

        return action, log_prob, value, new_hidden

    def evaluate_sequence(self, obs_sequence, done_sequence, action_sequence, init_hidden, return_std: float):
        """Used during training (BPTT). Evaluates a full trajectory sequence for a single environment."""

        def scan_step(carry_hidden, step_data):
            obs, done, action = step_data
           
            x = jax.nn.tanh(self.encoder(obs))
            
            # Boundary initialization
            z_init = jax.nn.tanh(self.I_network(x)) *0.
            current_hidden = jnp.where(done, z_init, carry_hidden)

            #  NCO loss (co-state target)
            def get_value_fn(obs_feat, h_state):
                h_next = self.ctrnn(obs_feat, h_state)
                return self.critic_out(h_next).squeeze()
            
            value = get_value_fn(x, current_hidden)
            
            lambda_target_scaled = jax.lax.stop_gradient(jax.grad(get_value_fn, argnums=0)(x, current_hidden))
            lambda_target = lambda_target_scaled * return_std

            # Cosine similarity 
            eps = 1e-8
            dot_product = jnp.sum(current_hidden * lambda_target)
            norm_z = jnp.sqrt(jnp.sum(jnp.square(current_hidden)) + eps)
            norm_lambda = jnp.sqrt(jnp.sum(jnp.square(lambda_target)) + eps)
            
            cos_sim = dot_product / (norm_z * norm_lambda)
            pmp_loss = 1.0 - cos_sim

            new_hidden = self.ctrnn(x, current_hidden)
            mean = jax.nn.tanh(self.actor_out(new_hidden)) * self.action_scale
        
            std = jnp.exp(self.log_std)
            var = std ** 2
            
            log_prob = -0.5 * ((action - mean) ** 2) / var - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
            log_prob = jnp.sum(log_prob) 
            entropy = jnp.sum(0.5 + 0.5 * jnp.log(2 * jnp.pi * var))
            
            return new_hidden, (value, log_prob, entropy, pmp_loss)

        _, (values, log_probs, entropies, pmp_losses) = jax.lax.scan(
            scan_step, init_hidden, (obs_sequence, done_sequence, action_sequence)
        )
        
        return values, log_probs, entropies, pmp_losses

    def get_value(self, obs, done, hidden):
        """Helper to get V(s) for GAE calculation."""
        x = jax.nn.tanh(self.encoder(obs))
        
        z_init = jax.nn.tanh(self.I_network(x))*0. 
        hidden = jnp.where(done, z_init, hidden)
        
        new_hidden = self.ctrnn(x, hidden)
        value = self.critic_out(new_hidden).squeeze()
        return value, new_hidden