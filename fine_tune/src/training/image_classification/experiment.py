import torch
import util

import data
import models
import training

def run_experiment():
  model = models.cifar10.get_model(depth=16, width=4, dropout_rate=0.0)
  loss = torch.nn.CrossEntropyLoss()
  optimizer_fn = util.bind(torch.optim.Adam, ..., lr=0.1, momentum=0.9, weight_decay=5e-4)

  trainset, valset, testset = data.cifar10.get_datasets()

  model = training.train(trainset=trainset, model=model, loss_fn=loss, optimizer_fn=optimizer_fn,
                         epochs=1, splits=1, batch_size=64, max_steps=10)