import pandas as pd
import matplotlib.pyplot as plt

from datautils import read_csv, remove_missing_rows, dummify_data
from plothelpers import plot_correlations

from sklearn.model_selection import train_test_split


def compute_correlations(X_train_df, y_train_df, label_col_str):
    '''
    Compute correlations between all features, select the column corresponding 
    to the label, sort values in descending order and drop the entry for the 
    label (correlation with itself)

    '''
    data_df = pd.concat([X_train, y_train_df], axis=1)
    correlations_with_target = data_df.corr()[label_col_str].sort_values(ascending=False)[1:]
    return correlations_with_target.to_frame()


if __name__ == '__main__':

    bank_raw_df = read_csv('data/bank-full.csv', delimiter_str=';')

    bank_data_df = remove_missing_rows(bank_raw_df)

    bank_features = bank_data_df.drop(columns=['y'])
    bank_labels = bank_data_df['y']

    bank_features_dummified = dummify_data(bank_features)

    print('Shape of features before dummification: ', bank_features.shape)
    print('Shape of features after dummification: ', bank_features_dummified.shape)

    X_train, X_test, y_train, y_test = train_test_split(bank_features_dummified,
                                                        bank_labels, 
                                                        test_size=0.2, 
                                                        random_state=20130810)

    y_train_df = pd.DataFrame(y_train.apply(lambda x: int(x == 'yes')), 
                              columns=['y'])

    corr_df = compute_correlations(X_train, y_train_df, 'y')
    corr_df.index.name = 'feature'
    corr_df.reset_index(inplace=True)

    plot_correlations(corr_df, feature_str='feature', label_str='y')
    plt.tight_layout()

    plt.show()
