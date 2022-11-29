import torch

import data
import models

model = models.cifar10.get_model(depth=16, width=4, dropout_rate=0.0)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

