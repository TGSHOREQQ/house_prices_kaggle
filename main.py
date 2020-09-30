import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# set pycharm terminal wider than default
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 22)

train_data = pd.read_csv('data/train.csv')

# skewness and kurtosis
print("Skewness: %f" % train_data['SalePrice'].skew())
print("Kurtosis: %f" % train_data['SalePrice'].kurt())

# Reduced features initially
features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Utilities', 'Neighborhood', 'OverallQual',
            'SalePrice']  # ,
# 'OverallCond', 'YearBuilt', 'RoofStyle', 'Foundation', 'BsmtQual', 'Heating', '1stFlrSF', '2ndFlrSF',
# 'KitchenQual', 'GarageType', 'GarageArea', 'PavedDrive', 'PoolQC', 'Fence', 'SaleType', 'SaleCondition', 'SalePrice']

train_data = train_data[features]


# function to plot all variables against target feature
def plot_variables(dataset, target):
    num_features = dataset.columns.value_counts().sum()
    print(dataset.info(0))
    numeric = []
    categorical = []
    for feature in dataset:
        if dataset[feature].dtype == 'object':
            categorical.append(feature)
        if dataset[feature].dtype == 'int64' or 'float64':
            numeric.append(feature)

    numeric.pop(-1)
    print('Numeric Features: %s' % numeric)
    print('Categorical Features: %s' % categorical)
    # Plot numeric features: Histogram and pairplots
    for feature in numeric:
        plt.hist(train_data[feature], bins=60)
        plt.xlabel(feature)
        plt.grid(True)
        plt.show()
    sns.pairplot
    # Bar plot
    #for feature in numeric:
    #    plt.hist(train_data[feature], bins=60)
    #    plt.show()



plot_variables(train_data, ['SalePrice'])
