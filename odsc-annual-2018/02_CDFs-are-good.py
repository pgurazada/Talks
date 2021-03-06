import matplotlib.pyplot as plt
import sys

from datautils import read_excel, remove_missing_rows, remove_redundant_features
from plothelpers import plot_cdf, plot_cdf_and_normal

from sklearn.model_selection import train_test_split


'''
https://www.andata.at/en/software-blog-reader/why-we-love-the-cdf-and-do-not-like-histograms-that-much.html

'''

if __name__ == '__main__':
      
    churn_raw_df = read_excel('data/WA_Fn-UseC_-Telco-Customer-Churn.xlsx', 
                               na_values_str=' ')

    churn_data_df = remove_redundant_features(churn_raw_df, ['customerID'])

    churn_data_df = remove_missing_rows(churn_data_df)

    churn_features = churn_data_df.drop(columns=['Churn'])
    churn_labels = churn_data_df['Churn']

    pt = PowerTransformer()

    X_train, X_test, y_train, y_test = train_test_split(churn_features,
                                                        churn_labels,
                                                        test_size=0.2,
                                                        random_state=20130810)

    if sys.argv[1] == 'ex1':
        plot_cdf(data_df=X_train, 
                 feature_label='tenure')

        plt.show()

    if sys.argv[1] == 'ex2':

        plot_cdf(data_df=X_train,
             feature_label='TotalCharges')

        plt.tight_layout()

        plt.show()

    if sys.argv[1] == 'ex3':

        plot_cdf_and_normal(data_df=X_train,
                            feature_label='TotalCharges')
    
        plt.tight_layout()

        plt.show()

