import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style('ticks')
plt.rcParams['figure.figsize'] = (12, 8)


def plot_histogram(data_df, feature_label, nb_bins=10):

    plt.hist(data_df[feature_label], 
             bins=nb_bins, 
             color='black',
             edgecolor='white',
             linewidth=1.2)

    plt.xlabel(feature_label)
    plt.title('Distribution of ' + feature_label + ' (bins = ' + str(nb_bins) + ')')


def plot_cdf(data_df, feature_label):

    x = np.sort(data_df[feature_label])
    y = np.linspace(0, 1, len(x), endpoint=False)

    plt.plot(x, y, color='black')
    for q in [0.25, 0.5, 0.75]:
        plt.axvline(x=np.quantile(x, q),
                    alpha=0.8, 
                    linewidth=1.2)

        plt.axhline(y=q, 
                    alpha=0.8, 
                    linewidth=1.2)

    plt.xlabel(feature_label)
    plt.ylabel('F(x)')
    plt.title('CDF of ' + feature_label)


def plot_cdf_and_normal(data_df, feature_label):
    x = np.sort(data_df[feature_label])
    y = np.linspace(0, 1, len(x), endpoint=False)
    y_norm = ss.norm.cdf(x, np.mean(x), np.std(x))

    plt.plot(x, y, label='data')
    plt.plot(x, y_norm, label='normal distribution')
    
    plt.xlabel(feature_label)
    plt.legend()
    

def plot_correlations(correlation_df, feature_str, label_str):
    g = sns.PairGrid(correlation_df, x_vars=label_str, y_vars=feature_str,
                     size=18, aspect=.5)
    g.map(sns.stripplot, palette="viridis",
          size=6, orient='h', linewidth=1, edgecolor="white")

