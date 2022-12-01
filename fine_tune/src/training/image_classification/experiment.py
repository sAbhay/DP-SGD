import torch

from .util import bind
from .data.cifar10 import get_cifar10_data
from .models.cifar10 import get_model
from .training import train

def run_experiment():
  model = get_model(depth=16, width=4, dropout_rate=0.0)
  loss = torch.nn.CrossEntropyLoss()
  optimizer_fn = bind(torch.optim.Adam, ...)

  trainset, valset, testset = get_cifar10_data()

  model = train(trainset=trainset, model=model, loss_fn=loss, optimizer_fn=optimizer_fn,
                         epochs=1, splits=1, batch_size=64, max_steps=10)