import pandas as pd


def read_excel(data_path, na_values_str):
    '''
    Input should be an excel file
    '''

    data_raw_df = pd.read_excel(data_path, na_values=na_values_str)
    print('data set imported with the following', len(data_raw_df.columns), 'columns')
    print(data_raw_df.dtypes)

    return data_raw_df


def read_csv(data_path, delimiter_str):
    data_raw_df = pd.read_csv(data_path, delimiter=delimiter_str)
    print('data set imported with the following', len(data_raw_df.columns), 'columns')
    print(data_raw_df.dtypes)

    return data_raw_df


def remove_redundant_features(raw_data_df, cols_to_drop):
    return raw_data_df.drop(columns=cols_to_drop)


def remove_missing_rows(data_df):
    return data_df.dropna()


def dummify_data(data_df, dummy_na_flag=False):
    return pd.get_dummies(data_df, dummy_na=dummy_na_flag)

