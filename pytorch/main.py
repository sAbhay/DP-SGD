import torch.optim

import data
from torch import nn

import experiment
from dp_sgd_optimizer import DPSGD
import loss

import pandas as pd
import matplotlib.pyplot as plt

d = 512
N = 32
h = d
batch_size = 32
epochs = 20
grad_norm_bound = 1
noise_scale = 0.5
lr = 5e-2

X, Y = data.generate_noisy_data(d, N, delta=0.01)
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
    nn.Linear(h, h),
    nn.Sigmoid(),
    nn.Linear(h, h),
    nn.Sigmoid(),
    nn.Linear(h, h),
    nn.Sigmoid(),
    nn.Linear(h, 1)
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            nn.init.xavier_uniform_(m.weight)
            m.bias.fill_(0.)

model.apply(init_weights)

results = experiment.run_experiments(model, train_funcs, optimizers, train_loader, test_loader, X, Y, loss_functions, epochs)
grad_l2s = {}
for key, result in results.items():
    if result["grad_l2s"] is not None:
        grad_l2s[key] = result["grad_l2s"]
        result["grad_l2s"] = []
print(results)

for k, norms in grad_l2s.items():
    df = pd.DataFrame(norms[-1], columns=['accurate', 'norm'])
    ax = df.plot.hist(column=['norm'], by='accurate', sharey=True, sharex=True)
    plt.show()

# print("Gap:", dpsgd_loss - sgd_loss, loss_name, "DP-SGD Loss:", dpsgd_loss, "SGD Loss:", sgd_loss,
#           "As percent of DPSGD Loss:", (dpsgd_loss - sgd_loss) / dpsgd_loss)
