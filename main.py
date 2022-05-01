import copy

import torch.optim

import data
from torch import nn
from dp_sgd_optimizer import DPSGD

d = 128
N = 1024
batch_size = 32
epochs = 20
grad_norm_bound = 1
noise_scale = 0.5
lr = 1e-1


X, Y = data.generate_perfect_data(d, N)
train_batch_loader, train_sample_loader, test_loader = data.load_tensor_to_dataloader(X, Y, batch_size=batch_size)

loss_functions = {"BCE": nn.BCELoss(), "Hinge": nn.HingeEmbeddingLoss()}

for loss_name, loss_function in loss_functions.items():
    dp_model = nn.Sequential(
        nn.Linear(d, 1),
        nn.Sigmoid()
    )

    model = copy.deepcopy(dp_model)

    optimizer = DPSGD(
        params=dp_model.parameters(), lr=lr, noise_scale=noise_scale, group_size=batch_size,
        grad_norm_bound=grad_norm_bound,
    )

    sgd = torch.optim.SGD(model.parameters(), lr=lr)

    assert(loss_function(model(X), Y) - loss_function(dp_model(X), Y) == 0)

    for e in range(epochs):
        for X_batch, y_batch in train_batch_loader:
            sgd.zero_grad()
            loss = loss_function(model(X_batch), y_batch)
            loss.backward()
            sgd.step()
        sgd_loss = loss_function(model(X), Y)
    print("SGD", loss_name, e, sgd_loss)

    for e in range(epochs):
        for X_batch, y_batch in train_batch_loader:
            optimizer.zero_grad()
            for x, y in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_batch, y_batch)):
                y_pred = dp_model(x)
                loss = loss_function(y_pred, y)
                loss.backward()
                optimizer.per_sample_gradient_clip()
            # loss = loss_function(model(X_batch), y_batch)
            # loss.backward()
            optimizer.step()
        dpsgd_loss = loss_function(dp_model(X), Y)
    print("DPSGD", loss_name, e, dpsgd_loss)

    print("Gap:", dpsgd_loss-sgd_loss, loss_name, "DP-SGD Loss:", dpsgd_loss, "SGD Loss:", sgd_loss, "As percent of DPSGD Loss:", (dpsgd_loss-sgd_loss)/dpsgd_loss)
