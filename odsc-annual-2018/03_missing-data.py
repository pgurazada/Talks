import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context('talk')
sns.set_style('ticks')
plt.rcParams['figure.figsize'] = (12, 6)


warnings.filterwarnings(action='ignore', category=FutureWarning)


class MissingDataInformer:

    def __init__(self, n=10):
        self.max_features = n
    
    def compute_value_counts(self, data_df):
        return data_df.isnull().sum().sort_values(ascending=False)

    def compute_value_perc(self, data_df):
        return data_df.isnull().sum().sort_values(ascending=False) * 100/data_df.shape[0]

    def cols_with_missings(self, data_df):
        return [col for col in data_df.columns if data_df[col].isnull().any()]
        

if __name__ == '__main__':

    ufo_df = pd.read_feather('data/ufo.feather')
    print(ufo_df.shape)

    mdi = MissingDataInformer()

    print(mdi.compute_value_counts(ufo_df))
    print(mdi.compute_value_perc(ufo_df))
    print(mdi.cols_with_missings(ufo_df))
