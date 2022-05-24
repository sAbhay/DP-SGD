import pandas as pd
import _pickle as cPickle
import matplotlib.pyplot as plt
import seaborn as sns


with open('grad_norms_dpsgd.pkl.pbz2', 'rb') as f:
    epoch_norms = cPickle.load(f)

norms = []
for epoch in range(len(epoch_norms)):
    norms += [(epoch,)+v for v in epoch_norms[epoch]]
for i, norm in enumerate(norms):
    norms[i] = (norm[0], norm[1].item(), norm[2].item(), *norms[3])

with open('param_norms_dpsgd.pkl.pbz2', 'rb') as f:
    param_norms = cPickle.load(f)
for i, param_norm in enumerate(param_norms):
    param_norms[i] = (i, param_norm)

cols = ['epoch', 'norm', 'accurate']
logit_cols = [f'{i}_logit' for i in range(0, 10)]
columns = cols + logit_cols
sample_df = pd.DataFrame(norms, columns=columns)
sample_df['logits_stddev'] = sample_df[logit_cols].std(axis=1)

# print(len(df[df['epoch']==0]), len(df[(df['epoch']==0) & (df['accurate']==0)]), len(df[(df['epoch']==0) & (df['accurate']==1)]), len(df[df['epoch']==10]))
# ax = df.hist(column=['norm'], by=['epoch'], sharey=True, sharex=True)
sns.histplot(sample_df, x='epoch', y='norm', hue='accurate')
sns.scatterplot(sample_df, x='norm', y='logits_stddev', c='epoch', style='accurate')

cols = ['epoch', 'param_norm']
epoch_df = pd.DataFrame(param_norms, columns=cols)
epoch_df['expected_grad_norm'] = sample_df.groupby(['epoch']).mean()
sns.lineplot(x='epoch', y='value', hue='variable',
             data=pd.melt(epoch_df[['epoch', 'param_norm', 'expected_grad_norm']], ['epoch']))

plt.show()