import torch

from .util import bind
from .data.cifar10 import get_cifar10_data
from .models.cifar10 import get_model
from .training import train
from .evaluation import total_loss, accuracy

def run_experiment():
  model = get_model(depth=16, width=4, dropout_rate=0.0).cuda()

  loss = torch.nn.CrossEntropyLoss()
  optimizer_fn = bind(torch.optim.SGD, ..., lr=0.1)

  trainset, valset, testset = get_cifar10_data()
  print(f"Dataset sizes: train: {len(trainset)}, val: {len(valset)}, test: {len(testset)}")

  model = train(trainset=trainset, model=model, loss_fn=loss, optimizer_fn=optimizer_fn,
                         epochs=600, splits=2500, batch_size='full_batch', max_steps=4, valset=valset, max_dist=12,
                noise_multiplier=0.1)
  print(f"Final train loss: {total_loss(model, loss, trainset)}, accuracy: {accuracy(model, trainset)}")
  print(f"Final val loss: {total_loss(model, loss, valset)}, accuracy: {accuracy(model, valset)}")