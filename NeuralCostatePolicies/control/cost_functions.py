from abc import ABC
from dataclasses import dataclass
import jax.numpy as jnp
from jaxtyping import Float, Num
from jaxtyping import Array, Float
import equinox as eqx

# @dataclass 
class QuadraticCost(eqx.Module):
    """
        Quadratic cost function L(x,u) = x^T Q x + u^T R u, for a single state. 
        Can be integrated as part of the ODE to obtain the trajectory cost.

        Parameters: 
            Q (Array) State cost matrix of shape (D_sys,D_sys). Typically identity matrix with scaled diagonal. Should be PSD.
            R (Array) Control cost matrix of shape (D_control, D_control). Should be PSD.
    """
    Q:Array = None
    R:Array = None
    x_star:Array = None
    u_star:Array = None
    transform:callable=None

    def __call__(self, x:Array, u:Array):
        if self.transform is not None:
            x = self.transform(x)
        x = x-self.x_star
        u = u-self.u_star
        return x.T@self.Q@x + u.T@self.R@u

    def __post_init__(self):
        if self.x_star is None:
            self.x_star = jnp.zeros((self.Q.shape[0]),)
        if self.u_star is None:
            self.u_star = jnp.zeros((self.R.shape[0]),)

        self._check_shapes()
        # self._check_PSD()

    def _check_shapes(self):
        assert self.Q.shape[0] == self.x_star.shape[0], f"Q and x* do not have the same dimensionality: Q shape: {self.Q.shape[0]} and x* shape: {self.x_star.shape[0]}"
        assert self.R.shape[0] == self.u_star.shape[0], f"R and u* do not have the same dimensionality: R shape: {self.R.shape[0]} and u* shape: {self.u_star.shape[0]}"

    # def _check_PSD(self):
    #     assert jnp.all(jnp.linalg.eigh(self.Q)>=0.), "Q is not positive-semidefinite matrix. "
    #     assert jnp.all(jnp.linalg.eigh(self.R)>=0.), "R is not positive-semidefinite matrix. " 


@dataclass
class HomotopyFuelCost(ABC):
    """
    Homotopy Cost function L(x,u) = ||u|| + epsilon  * ||u||^2.
    
    This serves as a bridge between Minimum Energy (Quadratic) and Minimum Fuel (Linear) problems.
    
    Parameters:
        epsilon (float): The homotopy parameter. 
                         High values (~1.0) -> Behaves like Min-Energy (easy to solve).
                         Low values (~1e-5) -> Behaves like Min-Fuel (Bang-Off-Bang).
        u_star (Array):  Nominal control to penalize deviation from. Defaults to zero vector.
        safe_tol (float): Small number to ensure numerical stability of gradients at u=0.
    """
    epsilon: float = 1.0
    u_star: Array = None
    R:Array = None
    safe_tol: float = 1e-12

    def __call__(self, x: Array, u: Array):
        # Shift control if a non-zero reference exists
        if self.u_star is not None:
            u = u - self.u_star

        # Compute squared magnitude: ||u||^2 = u^T u
        u_sq = u.T@self.R@u

        # Compute magnitude: ||u|| = sqrt(u^T u)
        u_norm = jnp.sqrt(u_sq + self.safe_tol)

        # Combine terms: Cost = ||u|| + epsilon * ||u||^2
        return u_norm + self.epsilon * u_sq

    def __post_init__(self):
        # If u_star is provided, ensure shapes make sense, otherwise ignore
        if self.u_star is not None:
            # Basic sanity check only if u_star is used
            pass


@dataclass
class QuadraticIntegralCost(ABC):
    """
    Quadratic control cost objective. No termination cost is assumed currently.       
    
    C(x,u) =  \int_{T-\Delta t}_t x(\tau) Q x(\tau) + u(\tau) R u(\tau) d\tau

    Args:
        Q (D_sys,)      Trajectory state cost matrix
        R (D_control,)  Control cost matrix
        Q (D_sys,)      Termination state cost matrix
        x_star (D_sys,) or (T, D_sys) optimal state.
        u_star (D_control,) or (T, D_control) optimal control input.
    """
    x_star:Array
    u_star:Array
    Q:Array=None
    R:Array=None
    Q_f:Array=None
    dt:Float = 1

    def __post_init__(self)->None:
        if self.Q is not None:
            assert self.x_star.shape[-1] == self.Q.shape[-1], f"Dimensionality of x* does not match state cost matrix: xs {self.x_star.shape} vs. Q: {self.Q.shape}"
        if self.R is not None:
            assert self.u_star.shape[-1] == self.R.shape[-1], f"Dimensionality of u* does not match control cost matrix: us {self.u_star.shape} vs. R: {self.R.shape}"
        if self.Q_f is not None:
            assert self.Q.shape == self.Q_f.shape, f"Dimensionality of Q and Q_f should be the same, but are instead: Q-{self.Q.shape} and Q_f-{self.Q_f.shape}"
        if self.Q_f is None and self.Q is not None:
            self.Q_f = jnp.copy(self.Q)

    def __call__(self, xs:Array, us:Array) -> Float:
        """
        Compute the control cost for a single trajectory x(t) given control input u(t).

        Args:
            xs (Array) (T,D_system) Integrated states over time
            us (Array) (T,D_system) Controls over time.

        Return:
            cost (Float) cost of the control sequence u(t)
        """
        self._check_inputs(xs, us)
        cost = self.dt*jnp.linalg.norm(self.Q @ (xs - self.x_star).T, axis=0).sum() if self.Q is not None else 0.
        cost += jnp.linalg.norm(self.Q_f @ (xs[-1:] - self.x_star).T, axis=0).sum() if self.Q_f is not None else 0.
        cost += self.dt*jnp.linalg.norm(self.R @ (us - self.u_star).T, axis=0).sum() if self.R is not None else 0.
        return cost
    
    def _check_inputs(self, xs:Array, us:Array):
        assert len(xs.shape) <= 2, "Assumed that the trajectories are from a single trial"
        if self.Q is not None:
            assert xs.shape[-1] == self.Q.shape[-1], f"Dimensionality of states xs does not match state cost matrix: xs {xs.shape} vs. Q: {self.Q.shape}"
        if self.R is not None:
            assert us.shape[-1] == self.R.shape[-1], f"Dimensionality of inputs us does not match control cost matrix: us {us.shape} vs. R: {self.R.shape}"
