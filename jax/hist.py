import pandas as pd
import pickle
import matplotlib.pyplot as plt

epoch_norms = []
with open('grad_norms.pkl', 'rb') as f:
    epoch_norms = pickle.load(f)

norms = []
for epoch in range(len(epoch_norms)):
    norms += [(epoch,)+v for v in epoch_norms[epoch]]
for i, norm in enumerate(norms):
    norms[i] = (norm[0], norm[1].item(), norm[2].item())

df = pd.DataFrame(norms, columns=['epoch', 'norm', 'accurate'])
# print(len(df[df['epoch']==0]), len(df[(df['epoch']==0) & (df['accurate']==0)]), len(df[(df['epoch']==0) & (df['accurate']==1)]), len(df[df['epoch']==10]))
ax = df.hist(column=['norm'], by=['epoch', 'accurate'], sharey=True, sharex=True)
plt.show()