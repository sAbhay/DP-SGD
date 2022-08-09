## From jax/jax/_src/util.py (https://github.com/google/jax/blob/36cd56824def5a6caac2cf1b837f676cfbc8dfb7/jax/_src/util.py)

def safe_zip(*args):
  n = len(args[0])
  for arg in args[1:]:
    assert len(arg) == n, 'length mismatch: {}'.format(list(map(len, args)))
  return list(zip(*args))

def safe_map(f, *args):
  args = list(map(list, args))
  n = len(args[0])
  for arg in args[1:]:
    assert len(arg) == n, 'length mismatch: {}'.format(list(map(len, args)))
  return list(map(f, *args))

def unzip2(xys):
  xs = []
  ys = []
  for x, y in xys:
    xs.append(x)
    ys.append(y)
  return tuple(xs), tuple(ys)


## Source: https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75

from jax import numpy as jnp
from jax.lib import pytree

def tree_stack(trees):
  """Takes a list of trees and stacks every corresponding leaf.
  For example, given two trees ((a, b), c) and ((a', b'), c'), returns
  ((stack(a, a'), stack(b, b')), stack(c, c')).
  Useful for turning a list of objects into something you can feed to a
  vmapped function.
  """
  leaves_list = []
  treedef_list = []
  for tree in trees:
    leaves, treedef = pytree.flatten(tree)
    leaves_list.append(leaves)
    treedef_list.append(treedef)

  grouped_leaves = zip(*leaves_list)
  result_leaves = [jnp.stack(l) for l in grouped_leaves]
  return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
  """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
  For example, given a tree ((a, b), c), where a, b, and c all have first
  dimension k, will make k trees
  [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
  Useful for turning the output of a vmapped function into normal objects.
  """
  leaves, treedef = pytree.flatten(tree)
  n_trees = leaves[0].shape[0]
  new_leaves = [[] for _ in range(n_trees)]
  for leaf in leaves:
    for i in range(n_trees):
      new_leaves[i].append(leaf[i])
  new_trees = [treedef.unflatten(l) for l in new_leaves]
  return new_trees