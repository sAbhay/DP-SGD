import torch
from functools import partial


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
  for key in sd:
    sd[key] += (torch.randn(sd[key].size()) * std_scalar).long().cuda()
  model.load_state_dict(sd)
  return model