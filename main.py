import torch.optim

import data
from torch import nn

import experiment
from dp_sgd_optimizer import DPSGD
import loss

d = 128
N = 1024
h = 128
batch_size = 32
epochs = 5
grad_norm_bound = 1
noise_scale = 0.5
lr = 1e-1

X, Y = data.generate_perfect_data(d, N)
train_loader, test_loader = data.load_tensor_to_dataloader(X, Y, batch_size=batch_size)

loss_functions = {"BCE": nn.BCEWithLogitsLoss()}
optimizers = {
    "SGD": (torch.optim.SGD, {"lr": lr}),
    "DPSGD": (DPSGD, {"lr": lr, "noise_scale": noise_scale, "group_size": batch_size,
                      "grad_norm_bound": grad_norm_bound})
}
train_funcs = {"SGD": experiment.train_sgd, "DPSGD": experiment.train_dpsgd}

model = nn.Sequential(
    nn.Linear(d, h),
    nn.Sigmoid(),
    nn.Linear(h, h),
    nn.Sigmoid(),
    nn.Linear(h, 1)
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

results = experiment.run_experiments(model, train_funcs, optimizers, train_loader, test_loader, X, Y, loss_functions, epochs)
print(results)

# print("Gap:", dpsgd_loss - sgd_loss, loss_name, "DP-SGD Loss:", dpsgd_loss, "SGD Loss:", sgd_loss,
#           "As percent of DPSGD Loss:", (dpsgd_loss - sgd_loss) / dpsgd_loss)
