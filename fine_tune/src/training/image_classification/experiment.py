import torch

from .util import bind
from .data.cifar10 import get_cifar10_data
from .models.cifar10 import get_model
from .training import train
from .evaluation import total_loss, accuracy

def run_experiment():
  model = get_model(depth=16, width=4, dropout_rate=0.0)
  loss = torch.nn.CrossEntropyLoss()
  optimizer_fn = bind(torch.optim.SGD, ..., lr=0.1)

  trainset, valset, testset = get_cifar10_data()
  print(f"Dataset sizes: train: {len(trainset)}, val: {len(valset)}, test: {len(testset)}")

  model = train(trainset=trainset, model=model, loss_fn=loss, optimizer_fn=optimizer_fn,
                         epochs=1, splits=1, batch_size=64, max_steps=50)
  print(f"Train loss: {total_loss(model, loss, trainset)}, accuracy: {accuracy(model, trainset)}")
  print(f"Val loss: {total_loss(model, loss, valset)}, accuracy: {accuracy(model, valset)}")