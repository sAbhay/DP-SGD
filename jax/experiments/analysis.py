import pandas as pd
import pickle
import matplotlib.pyplot as plt
import scipy.special
import seaborn as sns
import os


NORM_DIR = r'norms'

def get_hyperparameter_strings(norm_dir):
    norm_files = [f for f in os.listdir(norm_dir) if os.path.isfile(os.path.join(norm_dir, f))]
    hyperparameter_strings = []
    for f in norm_files:
        f = os.path.basename(f)
        splits = f.split('_')
        if len(splits) < 4:
            continue
        hyperparameter_string = '_'.join(splits[2:])
        hyperparameter_string = hyperparameter_string.replace(".pkl", "").replace(".zip", "")
        hyperparameter_strings.append(hyperparameter_string)
    hyperparameter_strings = set(hyperparameter_strings)
    return hyperparameter_strings


def make_plots(hyperparams_string, plot_dir, norm_dir):
    print(f"Plotting {hyperparams_string}")
    # hyperparams_string = "dpsgd_loss=cross-entropy,lr=0.25,op=True,nm=1.3,l2nc=1.5"

    plot_dir = rf'{plot_dir}\{hyperparams_string}'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    with open(rf'{norm_dir}\grad_norms_{hyperparams_string}.pkl', 'rb') as f:
        epoch_norms = pickle.load(f)
    # print(epoch_norms[0])

    norms = []
    for epoch in range(len(epoch_norms)):
        norms += [(epoch,) + v for v in epoch_norms[epoch]]
    for i, norm in enumerate(norms):
        norms[i] = tuple([norm[0], norm[1], norm[2]] + list(norm[3]))
    # print(norms[0])

    with open(rf'{norm_dir}\param_norms_{hyperparams_string}.pkl', 'rb') as f:
        param_norms = pickle.load(f)
    for i, param_norm in enumerate(param_norms):
        param_norms[i] = (i, param_norm)

    cols = ['epoch', 'norm', 'accurate']
    logit_cols = [f'{i}_logit' for i in range(0, 10)]
    columns = cols + logit_cols
    sample_df = pd.DataFrame(norms, columns=columns)
    sample_df[logit_cols] = pd.DataFrame(scipy.special.softmax(sample_df[logit_cols].to_numpy(dtype=float), axis=1),
                                         columns=logit_cols)
    sample_df["max_logit"] = sample_df[logit_cols].max(axis=1)
    # print(sample_df)

    # print(len(df[df['epoch']==0]), len(df[(df['epoch']==0) & (df['accurate']==0)]), len(df[(df['epoch']==0) & (df['accurate']==1)]), len(df[df['epoch']==10]))
    # ax = df.hist(column=['norm'], by=['epoch'], sharey=True, sharex=True)
    ax = sample_df[(sample_df['epoch'] == 0) | (sample_df['epoch'] == 19)][['epoch', 'norm', 'accurate']]. \
        hist(column='norm', by=['epoch', 'accurate'], legend=False)
    # ax = sns.histplot(data=sample_df[(sample_df['epoch'] == 0) | (sample_df['epoch'] == 19)],
    #                   x='norm', stat='count', hue='accurate', by=)
    plt.savefig(rf'{plot_dir}\epoch_20_grad_norms_accuracy.png')
    plt.close()
    print("Saved Epoch 20 grad norms hist at", plot_dir)

    ax = sns.histplot(data=sample_df, x='epoch', y='norm', hue='accurate')
    sns.move_legend(ax, "upper left")
    plt.savefig(rf'{plot_dir}\grad_norms_accuracy.png')
    plt.close()
    print("Saved grad norms hist")

    cols = ['epoch', 'param_norm']
    epoch_df = pd.DataFrame(param_norms, columns=cols)
    epoch_df['expected_grad_norm'] = sample_df.groupby(['epoch']).mean()['norm']
    sns.lineplot(x='epoch', y='value', hue='variable',
                 data=pd.melt(epoch_df[['epoch', 'param_norm', 'expected_grad_norm']], ['epoch']))
    plt.savefig(rf'{plot_dir}\grad_norms_param_norms.png')
    plt.close()
    print("Saved grad and param norms plot")

    sample_df = sample_df[sample_df['epoch'] == 19]
    temp_df = sample_df[sample_df['accurate'] == True]
    slope, intercept, r_value_classified, p_value, std_err = scipy.stats.linregress(temp_df['norm'],
                                                                                    temp_df['max_logit'])
    temp_df = sample_df[sample_df['accurate'] == False]
    slope, intercept, r_value_misclassified, p_value, std_err = scipy.stats.linregress(temp_df['norm'],
                                                                                       temp_df['max_logit'])
    sns.scatterplot(data=sample_df, x='norm', y='max_logit', hue='epoch', style='accurate')
    plt.text(0, 0.55, 'R(norm, max logit) given correct classification: {:.3f}'.format(r_value_classified),
             size='small')
    plt.text(0, 0.5, 'R(norm, max logit) given misclassification: {:.3f}'.format(r_value_misclassified), size='small')
    plt.savefig(rf'{plot_dir}\grad_norms_max_logit.png')
    plt.close()
    print("Saved grad norms vs max logit hist")


PLOTS_DIR = r'C:\Users\abhay\Documents\P-Lambda\plots'

if __name__ == '__main__':
    # hyperparameter_strings = get_hyperparameter_strings(NORM_DIR)
    # print(hyperparameter_strings)
    hyperparameter_strings = ['dpsgd_loss=cross-entropy,lr=0.25,op=False,nm=1.3,l2nc=1.5,grp=8,bs=1024,ws=True,mu=0.999,ess=0,pss=0,pa=True,aug=0,rf=True,rc=True',
                              'dpsgd_loss=cross-entropy,lr=0.25,op=False,nm=1.3,l2nc=1.5,grp=8,bs=1024,ws=True,mu=0.999,ess=0,pss=0,pa=True,aug=4,rf=True,rc=True']

    for hyperparams_string in hyperparameter_strings:
        make_plots(hyperparams_string, PLOTS_DIR, NORM_DIR)