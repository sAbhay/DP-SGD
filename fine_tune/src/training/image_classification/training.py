import torch
import copy

from .util import add_models, mult_model, add_Gaussian_noise_model
from .project_gradient_descent import project_model_dist_constraint, model_dist, interpolate_model
from .evaluation import total_loss, accuracy


MODELS_PER_GPU = 4

def sub_train_loop(trainloader, model, loss_fn, optimizer, max_steps, model_ref=None, max_dist=None):
  step = 1
  training = True
  while training:  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
      inputs = inputs.cuda()
      labels = labels.cuda()

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        if model_ref is not None and max_dist is not None:
          print("Original model dist = {}, step {}".format(model_dist(model, model_ref), step), max_dist)
          model = project_model_dist_constraint(model, model_ref, max_dist)
          print("Model dist = {}, step {}".format(model_dist(model, model_ref), step), max_dist)

      # print statistics
      running_loss += loss.item()
      # print(f'[{step + 1}] loss: {running_loss / len(trainloader.dataset):.6f}')
      step += 1
      if step > max_steps:
        training = False
        break

  # print('Finished Training')
  return model


def train(trainset, model, loss_fn, optimizer_fn, epochs, splits, batch_size, max_steps, valset=None, max_dist=None):
  for epoch in range(epochs):
    partitions = torch.utils.data.random_split(trainset, [len(trainset)//splits]*splits, generator=torch.Generator().manual_seed(42))
    # model = model.cpu()
    running_average_model = None
    running_model_dist = 0
    for partition in partitions:
      trainloader = torch.utils.data.DataLoader(partition, batch_size=batch_size, shuffle=True, num_workers=2)
      model_copy = copy.deepcopy(model).cuda()
      optimizer = optimizer_fn(model_copy.parameters())

      sub_model = sub_train_loop(trainloader, model_copy, loss_fn, optimizer, max_steps)
      # print("Sub model dist = {}".format(model_dist(sub_model, model)))
      if max_dist is not None:
        with torch.no_grad():
          sub_model = interpolate_model(sub_model, model, max_dist)
      # print("Interpolated sub model dist = {}".format(model_dist(sub_model, model)))
      running_model_dist += model_dist(model, sub_model)

      if running_average_model is None:
        running_average_model = sub_model
      else:
        running_average_model = add_models(running_average_model, sub_model)
    running_average_model = mult_model(running_average_model, 1. / splits)
    model = running_average_model
    if max_dist is not None:
      with torch.no_grad():
        model = add_Gaussian_noise_model(model, torch.sqrt(max_dist / splits))

    print(f"Train loss: {total_loss(model, loss_fn, trainset)}, accuracy: {accuracy(model, trainset)}")
    if valset is not None:
      print(f"Val loss: {total_loss(model, loss_fn, valset)}, accuracy: {accuracy(model, valset)}")
    print(f"Average submodel Euclidean distance: {running_model_dist / len(partitions)}")
    print(f"Epoch {epoch} done")
  return model
