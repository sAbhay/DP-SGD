from sys import path
path.append('../../../')
from common import util, log
logger = log.get_logger('optimizer')

from typing import NamedTuple, Callable, Tuple
import functools
from functools import partial

from jax.example_libraries.optimizers import Params, InitFn, UpdateFn, Step, State, Updates, Schedule
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node
from collections import namedtuple

map = util.safe_map
zip = util.safe_zip

OptimizerState = namedtuple("OptimizerState",
                            ["packed_state", "tree_def", "subtree_defs"])
register_pytree_node(
    OptimizerState,
    lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
    lambda data, xs: OptimizerState(xs[0], data[0], data[1]))  # type: ignore[index]

GetParamsFn = Callable[[OptimizerState], Params]
SetParamsFn = Callable[[Updates, OptimizerState], OptimizerState]

class Optimizer(NamedTuple):
  init_fn: InitFn
  update_fn: UpdateFn
  get_params_fn: GetParamsFn
  set_params_fn: SetParamsFn


def optimizer(opt_maker: Callable[...,
  Tuple[Callable[[Params], State],
        Callable[[Step, Updates, Params], Params],
        Callable[[State], Params],
        Callable[[Updates, Params], Params]]]) -> Callable[..., Optimizer]:
  """Decorator to make an optimizer defined for arrays generalize to containers.

  With this decorator, you can write init, update, and get_params functions that
  each operate only on single arrays, and convert them to corresponding
  functions that operate on pytrees of parameters. See the optimizers defined in
  optimizers.py for examples.

  Args:
    opt_maker: a function that returns an ``(init_fun, update_fun, get_params, set_params)``
      triple of functions that might only work with ndarrays, as per

      .. code-block:: haskell

          init_fun :: ndarray -> OptStatePytree ndarray
          update_fun :: OptStatePytree ndarray -> OptStatePytree ndarray
          get_params :: OptStatePytree ndarray -> ndarray
          set_params :: OptStatePytree ndarray -> OptStatePytree ndarray

  Returns:
    An ``(init_fun, update_fun, get_params)`` triple of functions that work on
    arbitrary pytrees, as per

    .. code-block:: haskell

          init_fun :: ParameterPytree ndarray -> OptimizerState
          update_fun :: OptimizerState -> OptimizerState
          get_params :: OptimizerState -> ParameterPytree ndarray
          set_params :: OptimizerState -> OptimizerState

    The OptimizerState pytree type used by the returned functions is isomorphic
    to ``ParameterPytree (OptStatePytree ndarray)``, but may store the state
    instead as e.g. a partially-flattened data structure for performance.
  """

  @functools.wraps(opt_maker)
  def tree_opt_maker(*args, **kwargs):
    init, update, get_params, set_params = opt_maker(*args, **kwargs)

    @functools.wraps(init)
    def tree_init(x0_tree):
      x0_flat, tree = tree_flatten(x0_tree)
      initial_states = [init(x0) for x0 in x0_flat]
      states_flat, subtrees = util.unzip2(map(tree_flatten, initial_states))
      return OptimizerState(states_flat, tree, subtrees)

    @functools.wraps(update)
    def tree_update(i, grad_tree, opt_state):
      states_flat, tree, subtrees = opt_state
      grad_flat, tree2 = tree_flatten(grad_tree)
      if tree2 != tree:
        msg = ("optimizer update function was passed a gradient tree that did "
               "not match the parameter tree structure with which it was "
               "initialized: parameter tree {} and grad tree {}.")
        raise TypeError(msg.format(tree, tree2))
      states = map(tree_unflatten, subtrees, states_flat)
      new_states = map(partial(update, i), grad_flat, states)
      new_states_flat, subtrees2 = util.unzip2(map(tree_flatten, new_states))
      for subtree, subtree2 in zip(subtrees, subtrees2):
        if subtree2 != subtree:
          msg = ("optimizer update function produced an output structure that "
                 "did not match its input structure: input {} and output {}.")
          raise TypeError(msg.format(subtree, subtree2))
      return OptimizerState(new_states_flat, tree, subtrees)

    @functools.wraps(get_params)
    def tree_get_params(opt_state):
      states_flat, tree, subtrees = opt_state
      states = map(tree_unflatten, subtrees, states_flat)
      params = map(get_params, states)
      return tree_unflatten(tree, params)

    @functools.wraps(set_params)
    def tree_set_params(new_tree, opt_state):
      # logger.info("Opt_state set_params: {}".format(opt_state))
      states_flat, tree, subtrees = opt_state
      new_flat, tree2 = tree_flatten(new_tree)
      if tree2 != tree:
        msg = ("optimizer update function was passed a new tree that did "
               "not match the parameter tree structure with which it was "
               "initialized: parameter tree {} and new tree {}.")
        raise TypeError(msg.format(tree, tree2))
      states = map(tree_unflatten, subtrees, states_flat)
      new_states = map(set_params, new_flat, states)
      new_states_flat, subtrees2 = util.unzip2(map(tree_flatten, new_states))
      for subtree, subtree2 in zip(subtrees, subtrees2):
        if subtree2 != subtree:
          msg = ("optimizer update function produced an output structure that "
                 "did not match its input structure: input {} and output {}.")
          raise TypeError(msg.format(subtree, subtree2))
      return OptimizerState(new_states_flat, tree, subtrees)

    return Optimizer(tree_init, tree_update, tree_get_params, tree_set_params)
  return tree_opt_maker