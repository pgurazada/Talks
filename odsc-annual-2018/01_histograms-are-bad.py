import matplotlib.pyplot as plt

from datautils import read_dataset, remove_missing_rows, remove_redundant_features
from plothelpers import plot_histogram

from sklearn.model_selection import train_test_split


'''
Exploratory analysis should reveal at a very minimum two things:

- The distribution of features and labels
- Sense on which features are likely to be important

In this script we tackle the distribution aspect of features and labels

In particular, we show that histograms are widely used but probably not a great
way to assess the skwe of a distribution

https://www.business-science.io/business/2017/11/28/customer_churn_analysis_keras.html
'''


if __name__ == '__main__':
      
    churn_raw_df = read_dataset('data/WA_Fn-UseC_-Telco-Customer-Churn.xlsx', 
                                na_values_str=' ')

    churn_data_df = remove_redundant_features(churn_raw_df, ['customerID'])

    print(churn_data_df.shape)

    churn_data_df = remove_missing_rows(churn_data_df)
    print(churn_data_df.shape)

    churn_features = churn_data_df.drop(columns=['Churn'])
    churn_labels = churn_data_df['Churn']

    print(churn_features.shape, churn_labels.shape)

    churn_features_train, churn_features_test, churn_labels_train, churn_labels_test = train_test_split(churn_features, churn_labels, test_size=0.2, random_state=20130810)

    plt.figure(1)

    for i, bins in enumerate([6, 10, 12, 20]):
        plot_pos = 220+i+1
        plt.subplot(plot_pos)
        plot_histogram(data_df=churn_features_train, 
                       feature_label='tenure',
                       nb_bins=bins)
    
    plt.tight_layout()
    plt.show()

    plt.figure(2)

    for i, bins in enumerate([6, 60, 100, 200]):
        plot_pos = 220+i+1
        plt.subplot(plot_pos)
        plot_histogram(data_df=churn_features_train,
                       feature_label='TotalCharges', 
                       nb_bins=bins)
    
    plt.tight_layout()
    plt.show()
