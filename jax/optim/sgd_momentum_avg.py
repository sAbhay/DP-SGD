from .optimizer import Optimizer, optimizer, logger
from jax.example_libraries.optimizers import make_schedule
import jax.numpy as jnp

@optimizer
def sgd_momentum(step_size, mass: float):
  """Construct optimizer triple for SGD with momentum.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
    mass: positive scalar representing the momentum coefficient.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    v0 = jnp.zeros_like(x0)
    return x0, v0
  def update(i, g, state):
    x, velocity = state
    velocity = mass * velocity + g
    x = x - step_size(i) * g  # running sgd without momentum
    return x, velocity
  def get_params(state):
    x, _ = state
    return x
  def set_params(new_params, old_params):
    # logger.info(f"x_new: {x_new}, x: {x}")
    return new_params
  return init, update, get_params, set_params