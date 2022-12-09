import torch
from functools import partial


def tensor_euclidean_norm(m1):
  return torch.sqrt(torch.sum(torch.square(m1)))


def model_norm(modelA):
  sdA = modelA.state_dict()

  norm = 0.0
  for key in sdA:
    norm += torch.square(tensor_euclidean_norm(sdA[key]))

  return torch.sqrt(norm)


def average_param_mag(model):
  sd = model.state_dict()
  total = 0.0
  param_count = 0
  for key in sd:
    total += torch.sum(torch.abs(sd[key]))
    param_count += sd[key].numel()
  return total / param_count


# from https://stackoverflow.com/a/66274908/10163133
class bind(partial):
  """
  An improved version of partial which accepts Ellipsis (...) as a placeholder
  """

  def __call__(self, *args, **keywords):
    keywords = {**self.keywords, **keywords}
    iargs = iter(args)
    args = (next(iargs) if arg is ... else arg for arg in self.args)
    return self.func(*args, *iargs, **keywords)


# TODO: JIT all functions below
# from https://discuss.pytorch.org/t/average-each-weight-of-two-models/77008/2
def add_models(modelA, modelB):
  sdA = modelA.state_dict()
  sdB = modelB.state_dict()

  # Add all parameters
  for key in sdA:
    sdB[key] = sdB[key] + sdA[key]

  modelA.load_state_dict(sdB)
  return modelA


def mult_model(model, scalar):
  sd = model.state_dict()

  for key in sd:
    sd[key] = sd[key] * scalar

  model.load_state_dict(sd)
  return model


def add_Gaussian_noise_model(model, std_scalar):
  sd = model.state_dict()
  noise_norm = 0.0
  for key in sd:
    noise = (torch.randn(sd[key].size()) * std_scalar).long()
    noise_norm += tensor_euclidean_norm(noise)
    sd[key] += noise.cuda()
  model.load_state_dict(sd)
  return model, noise_norm