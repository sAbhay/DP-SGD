import torch
import copy
from torch.utils.tensorboard import SummaryWriter

from .util import add_models, mult_model, add_Gaussian_noise_model, model_norm, average_param_mag
from .project_gradient_descent import project_model_dist_constraint, model_dist, interpolate_model
from .evaluation import total_loss, accuracy

from src.accounting.accountant import Accountant


MODELS_PER_GPU = 4


def write_scalars(writer, n_iter, train_loss, val_loss, train_acc, val_acc, avg_submodel_dist, model_norm, noise_norm, avg_param_norm, param_noise_std, dp_epsilon):
  writer.add_scalar('Loss/train', train_loss, n_iter)
  writer.add_scalar('Loss/test', val_loss, n_iter)
  writer.add_scalar('Accuracy/train', train_acc, n_iter)
  writer.add_scalar('Accuracy/test', val_acc, n_iter)
  writer.add_scalar('Avg submodel dist', avg_submodel_dist, n_iter)
  writer.add_scalar('Total_norm/model', model_norm, n_iter)
  writer.add_scalar('Total_norm/noise', noise_norm, n_iter)
  writer.add_scalar('Avg_norm/param', avg_param_norm, n_iter)
  writer.add_scalar('Avg_norm/noise_std', param_noise_std, n_iter)
  writer.add_scalar('DP/epsilon', dp_epsilon, n_iter)



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


def train(trainset, model, loss_fn, optimizer_fn, epochs, splits, batch_size, max_steps, valset=None, max_dist=None, noise_multiplier=1):
  writer = SummaryWriter()
  accountant = Accountant(clipping_norm=max_dist, std_relative=noise_multiplier, dp_epsilon=1, dp_delta=1e-5,
                          num_samples=splits, batch_size=splits)
  for epoch in range(1, epochs+1):
    partitions = torch.utils.data.random_split(trainset, [len(trainset)//splits]*splits, generator=torch.Generator().manual_seed(42))
    # model = model.cpu()
    if batch_size == 'full_batch':
      batch_size = len(partitions[0])
    running_average_model = None
    running_model_dist = 0
    for partition in partitions:
      trainloader = torch.utils.data.DataLoader(partition, batch_size=batch_size, shuffle=True, num_workers=2)
      model_copy = copy.deepcopy(model).cuda()
      optimizer = optimizer_fn(model_copy.parameters())

      sub_model = sub_train_loop(trainloader, model_copy, loss_fn, optimizer, max_steps)
      # print("Sub model dist = {}".format(model_dist(sub_model, model)))
      with torch.no_grad():
        if max_dist is not None:
          sub_model = interpolate_model(sub_model, model, max_dist)
        # print("Interpolated sub model dist = {}".format(model_dist(sub_model, model)))
        running_model_dist += model_dist(model, sub_model)

        if running_average_model is None:
          running_average_model = sub_model
        else:
          running_average_model = add_models(running_average_model, sub_model)
    with torch.no_grad():
      running_average_model = mult_model(running_average_model, 1. / splits)
      model = running_average_model
      if max_dist is not None:
        model, noise_norm = add_Gaussian_noise_model(model, std_scalar=noise_multiplier*max_dist/splits)
      # print(f"Train loss: {total_loss(model, loss_fn, trainset)}, accuracy: {accuracy(model, trainset)}")
      # if valset is not None:
      #   print(f"Val loss: {total_loss(model, loss_fn, valset)}, accuracy: {accuracy(model, valset)}")
      # print(f"Average submodel Euclidean distance: {running_model_dist / len(partitions)}")
      # print(f"Model params norm: {model_norm(model)}, noise norm: {noise_norm}")
      # print(f"Average param norm: {average_param_mag(model)}, param noise std: {noise_multiplier*max_dist/splits}")
      write_scalars(writer, n_iter=epoch,
                    train_loss=total_loss(model, loss_fn, trainset),
                    val_loss=total_loss(model, loss_fn, valset),
                    train_acc=accuracy(model, trainset),
                    val_acc=accuracy(model, valset),
                    avg_submodel_dist=running_model_dist / splits,
                    model_norm=model_norm(model),
                    noise_norm=noise_norm,
                    avg_param_norm=average_param_mag(model),
                    param_noise_std=noise_multiplier*max_dist/splits,
                    dp_epsilon=accountant.compute_current_epsilon(epoch))
      print(f"Epoch {epoch} done")
  return model
