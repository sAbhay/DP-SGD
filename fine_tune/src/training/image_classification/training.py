import torch
import copy

from .util import add_models, mult_model
from .project_gradient_descent import project_model_dist_constraint, model_dist


MODELS_PER_GPU = 4

def sub_train_loop(trainloader, model, loss_fn, optimizer, max_steps, model_ref=None, max_dist=None):
  for step in range(max_steps):  # loop over the dataset multiple times

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

      if model_ref is not None and max_dist is not None:
        model = project_model_dist_constraint(model, model_ref, max_dist)

      # print statistics
      running_loss += loss.item()
    print(f'[{step + 1}] loss: {running_loss / len(trainloader.dataset):.6f}')

  print('Finished Training')
  return model


def train(trainset, model, loss_fn, optimizer_fn, epochs, splits, batch_size, max_steps):
  for epoch in range(epochs):
    partitions = torch.utils.data.random_split(trainset, [len(trainset)//splits]*splits, generator=torch.Generator().manual_seed(42))
    # model = model.cpu()
    running_average_model = None
    for partition in partitions:
      trainloader = torch.utils.data.DataLoader(partition, batch_size=batch_size, shuffle=True, num_workers=2)
      model_copy = copy.deepcopy(model).cuda()
      optimizer = optimizer_fn(model_copy.parameters())

      sub_model = sub_train_loop(trainloader, model_copy, loss_fn, optimizer, max_steps)
      print(f"Model dist: {model_dist(model, sub_model)}")

      if running_average_model is None:
        running_average_model = sub_model
      else:
        running_average_model = add_models(running_average_model, sub_model)
    running_average_model = mult_model(running_average_model, 1. / splits)
    model = running_average_model
    print(f"Epoch {epoch} done")
  return model
