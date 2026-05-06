import jax
import jax.numpy as jnp
import jax.random as jr
from NeuralCostatePolicies.control.cost_functions import QuadraticCost
from abc import ABC
import diffrax as dfx
from typing import Any, Tuple
from jaxtyping import Array
from mujoco_playground._src.mjx_env import State

class AbstractEnv(ABC):
    """
    Standard interface for RL environments.
    """
    def reset(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, dict]:
        """
        Resets the environment.
        Returns: (observation, info)
        """
        pass

    def step(self, state: Any, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]:
        """
        Steps the environment by one control interval (dt).
        
        Args:
            state: The current internal state of the environment.
            action: Control input to apply for this interval.
            
        Returns:
            next_state: The updated state.
            observation: The observation (y) derived from the state.
            reward: The scalar reward (negative cost) for this step.
            done: Boolean flag if episode is over.
            info: Dictionary with extra data (e.g. true state if partially observed).
        """
        pass
    
    def dt(self) -> float:
        """The control timestep (dt)."""
        pass


class DiffraxEnv(AbstractEnv):
    def __init__(self, dfx_system, env_config):
        self.system:AbstractSystem = dfx_system
        self.env_config = env_config
        self.D_sys:int = env_config.D_sys
        self.D_obs:int = env_config.D_obs
        self.D_control:int = env_config.D_control
        self.y0:float = env_config.y0

        self.ub:float = env_config.ub
        self.lb:float = env_config.lb

        self.R, self.Q, self.Q_f = env_config.R, env_config.Q, env_config.Q_f
        x_star, u_star = env_config.x_star, env_config.u_star
        cost_transform = env_config.cost_transform
        self.state_cost:callable = QuadraticCost(Q=self.Q, R=self.R, x_star=x_star, u_star=u_star, transform=cost_transform)
        self.termination_cost:callable = QuadraticCost(Q=self.Q_f, R=self.R*0., x_star=x_star, u_star=u_star, transform=cost_transform)

        self.name:str = env_config.system_name + " control task"
        self.dt:float = env_config.dt
        self.dt_dense:float = env_config.dt_dense
        assert self.dt_dense <= self.dt, f"Dense solver step dt_dense ({self.dt_dense}) is assumed to be smaller than control step dt ({self.dt})."
        self.obs_noise:float = env_config.obs_noise 
        self.max_steps:int = env_config.max_steps 

        
    def reset(self, key:jr.PRNGKey, x0:Array=None):
        """
        Resets environment. 
        x0 can be passed explicitly or sampled if you add distribution logic here.
        """
        # if x0 is None:
        x0 = ((jr.uniform(key, shape=x0.shape)-0.5)*4) #raise ValueError("x0 must be provided for deterministic Diffrax systems")
            
        t0 = jnp.array(0.0)                    
        state = (x0, t0)

        # Initial observation
        obs = self.observation(x0, key)
        
        return state, obs, {}
    
    def step(self, state:Array, action:Array, key:jr.PRNGKey=None):
        x_current, t_current = state
    
        t_next = t_current + self.dt        
        term = dfx.ODETerm(self._dynamics)
        
        sol = dfx.diffeqsolve(
            term, 
            self.system.solver, 
            t0=t_current, 
            t1=t_next, 
            dt0=self.dt_dense, 
            y0=x_current, 
            args=action
        )
        x_next = sol.ys[-1]

        # t_final = self.env_config.max_steps * self.dt
        done = False #t_next >= (t_final - 1e-5)

        state_cost = self.state_cost(x_next, action)
        # termination_cost = self.termination_cost(x_next, action)
        total_cost = state_cost #+ jnp.where(done, termination_cost, 0.0)
        obs = self.observation(x_next, key)
            
        new_state = (x_next, t_next)
        return new_state, obs, total_cost, done, {}
    
    def observation(self, x:jnp.ndarray, key:jr.PRNGKey=None):
        observation_function = self.env_config.observation
        return observation_function(x, key) if observation_function is not None else x

    def _dynamics(self, t:float, y:Array, args):
        return self.system.f(t, y, args)
    

class MujocoPlaygroundEnv(AbstractEnv):
    def __init__(self, mujoco_playground_env, env_config):
        self.mjx_env = mujoco_playground_env
        self.env_config = env_config
        self.D_sys:int = env_config.D_sys
        self.D_obs:int = env_config.D_obs 
        self.D_control:int = env_config.D_control

        self.ub:float = env_config.ub
        self.lb:float = env_config.lb

        self.R, self.Q, self.Q_f = env_config.R, env_config.Q, env_config.Q_f
        x_star, u_star = env_config.x_star, env_config.u_star
        cost_transform = env_config.cost_transform
        self.state_cost:callable = QuadraticCost(Q=self.Q, R=self.R, x_star=x_star, u_star=u_star, transform=cost_transform)
        self.termination_cost:callable = QuadraticCost(Q=self.Q_f, R=self.R*0., x_star=x_star, u_star=u_star, transform=cost_transform)
        self.use_mujoco_cost:bool = env_config.use_mujoco_cost

        self.name:str = env_config.system_name + " control task"
        self.dt:float = env_config.dt
        self.max_steps:int = env_config.max_steps

        self._jit_reset = jax.jit(self.mjx_env.reset)
        self._jit_step = jax.jit(self.mjx_env.step)

    def reset(self, key:jr.PRNGKey):
        """ Resets environment. """
        next_state = self._jit_reset(key)
        obs = self.observation(next_state.obs, key) # manually overwrite to design partially observable experiments in Mujoco Playground.
        return next_state, obs, next_state.metrics

    def step(self, state:State, action:Array, key:jr.PRNGKey=None):
        """ Take an environment step"""
        next_state = self._jit_step(state, action)
        reward = next_state.reward
        obs = self.observation(next_state.obs, key) # manually overwrite to design partially observable experiments in Mujoco Playground.
        cost = -reward if self.use_mujoco_cost else self.state_cost(obs, action)
        return next_state, obs, cost, next_state.done, next_state.info  

    def observation(self, x:jnp.ndarray, key:jr.PRNGKey=None):
        observation_function = self.env_config.observation
        return observation_function(x, key) if observation_function is not None else x     

    def dt(self) -> float:
        """
        Returns the control timestep.
        In Brax/MJX, this is typically available directly on the environment object.
        """
        return self.mjx_env.dt

    def _dynamics(self, t:float, y:Array, args):
        raise NotImplementedError("Not available in MJX environment.")
    
    def render(self, rollout_states:list):
        return self.env.render(rollout_states)