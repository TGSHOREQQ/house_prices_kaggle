import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# set pycharm terminal wider than default
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 22)



train_data = pd.read_csv('data/train.csv')


#print(train_data)

#print(train_data['SalePrice'].describe())

# plt.hist(train_data['SalePrice'], bins=60)
# plt.show()

#skewness and kurtosis
#print("Skewness: %f" % train_data['SalePrice'].skew())
#print("Kurtosis: %f" % train_data['SalePrice'].kurt())



#scatter plot grlivarea/saleprice
# variables = ['GrLivArea', 'TotalBsmtSF']
# for var in variables:
#     data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
#     data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# plt.show()


features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Utilities', 'Neighborhood', 'OverallQual',
 'OverallCond', 'YearBuilt', 'RoofStyle', 'Foundation', 'BsmtQual', 'Heating', '1stFlrSF', '2ndFlrSF', 'KitchenQual',
            'GarageType', 'GarageArea', 'PavedDrive', 'PoolQC', 'Fence', 'SaleType', 'SaleCondition', 'SalePrice']

train_data = train_data[features]

# function to plot all variables against target feature
def plot_variables(dataset, target):
    num_features = dataset.columns.value_counts().sum()
    fig = plt.figure(figsize=(14, 4))
    (ax1, ax2, ax3) = fig.subplots(1, num_features)
    fig.suptitle('Variables EDA', size=16)

    numeric = []
    categorical = []
    for feature in dataset:
        if dataset[feature].dtype.name == 'int64' or 'float64':
            numeric.append(feature)
        if dataset[feature].dtype.name == 'object':
            categorical.append(feature)

    numeric.pop(-1)
    print(numeric)
        #f'ax{val}' = 1#fig.subplots(1, 3)
    plt.show()
    # for var in numeric:
    #      data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
    #      data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
    # plt.show()

plot_variables(train_data, train_data['SalePrice'])

#dtype_mapping = dict(train_data.dtypes)
#print(dtype_mapping)

#numeric_cols = [ c for c in cols if dtype_mapping[c] != 'string' ]
#print(train_data.loc[:,train_data.dtypes==np.object], '\n')

#print(train_data.loc[:,train_data.dtypes!=np.object])

#plot_variables(train_data)