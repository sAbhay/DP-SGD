import torch


def tensor_euclidean_dist(m1, m2):
  return torch.sum(torch.square(m1 - m2))


def model_dist(modelA, modelB):
  sdA = modelA.state_dict()
  sdB = modelB.state_dict()

  dist = 0
  for key in sdA:
    dist += tensor_euclidean_dist(sdB[key], sdA[key])

  return dist


def project_model_dist_constraint(model, model_ref, max_dist, dist=None):
  if dist is None:
    dist = model_dist(model, model_ref)
  if dist < max_dist:
    return model

  sd = model.state_dict()
  sd_ref = model_ref.state_dict()

  for key in sd:
    sd[key] = (sd[key] - sd_ref[key]) * max_dist / model_dist(model, model_ref)

  model.load_state_dict(sd)
  return model
