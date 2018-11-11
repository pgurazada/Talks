import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

sns.set_context('talk')
sns.set_style('ticks')
plt.rcParams['figure.figsize'] = (12, 6)

'''
Exploratory analysis should reveal at a very minimum two things:

- The distribution of features and labels
- Sense on which features are likely to be important

https://www.business-science.io/business/2017/11/28/customer_churn_analysis_keras.html
'''

def read_dataset(data_path):
    '''
    Input should be an excel file
    '''

    data_raw_df = pd.read_excel(data_path)
    print('data set imported with the following',
          len(data_raw_df.columns),
          'columns')
    print(data_raw_df.columns)
    print(data_raw_df.dtypes)

    return data_raw_df


def remove_redundant_features(raw_data_df, cols_to_drop):
      return raw_data_df.drop(columns=cols_to_drop)



def remove_missing_rows(data_df):
      return data_df.dropna()


if __name__ == '__main__':

      churn_raw_df = read_dataset('data/WA_Fn-UseC_-Telco-Customer-Churn.xlsx')

      churn_data_df = remove_redundant_features(churn_raw_df, 
                                                ['customerID'])

      churn_data_df = remove_missing_rows(churn_data_df)







