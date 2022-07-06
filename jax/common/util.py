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