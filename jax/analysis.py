import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


with open('grad_norms_dpsgd.pkl', 'rb') as f:
    epoch_norms = pickle.load(f)
# print(epoch_norms[0])

norms = []
for epoch in range(len(epoch_norms)):
    norms += [(epoch,)+v for v in epoch_norms[epoch]]
for i, norm in enumerate(norms):
    norms[i] = tuple([norm[0], norm[1].item(), norm[2].item()] + list(norm[3]))
# print(norms[0])

with open('param_norms_dpsgd.pkl', 'rb') as f:
    param_norms = pickle.load(f)
for i, param_norm in enumerate(param_norms):
    param_norms[i] = (i, param_norm.item())

cols = ['epoch', 'norm', 'accurate']
logit_cols = [f'{i}_logit' for i in range(0, 10)]
columns = cols + logit_cols
sample_df = pd.DataFrame(norms, columns=columns)
sample_df['logits_stddev'] = sample_df[logit_cols].std(axis=1)
# print(sample_df)

# print(len(df[df['epoch']==0]), len(df[(df['epoch']==0) & (df['accurate']==0)]), len(df[(df['epoch']==0) & (df['accurate']==1)]), len(df[df['epoch']==10]))
# ax = df.hist(column=['norm'], by=['epoch'], sharey=True, sharex=True)
ax = sns.histplot(data=sample_df, x='epoch', y='norm', hue='accurate')
sns.move_legend(ax, "upper left")
plt.show()

sns.scatterplot(data=sample_df, x='norm', y='logits_stddev', hue='epoch', style='accurate')
plt.show()

cols = ['epoch', 'param_norm']
epoch_df = pd.DataFrame(param_norms, columns=cols)
epoch_df['expected_grad_norm'] = sample_df.groupby(['epoch']).mean()['norm']
sns.lineplot(x='epoch', y='value', hue='variable',
             data=pd.melt(epoch_df[['epoch', 'param_norm', 'expected_grad_norm']], ['epoch']))
plt.show()