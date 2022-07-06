from jax.example_libraries.optimizers import Optimizer, optimizer, make_schedule

@optimizer
def sgd_avg(step_size, average_params):
  """Construct optimizer triple for stochastic gradient descent.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    return x0
  def update(i, g, x, *avg_args):
    x = x - step_size(i) * g
    return average_params(x, *avg_args)
  def get_params(x):
    return x
  return Optimizer(init, update, get_params)