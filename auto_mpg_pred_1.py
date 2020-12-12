import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# defining the column names
cols = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
# reading the .data file using pandas
df = pd.read_csv('./auto-mpg.data', names=cols, na_values = "?",
                comment = '\t',
                sep= " ",
                skipinitialspace=True)
#making a copy of the dataframe
data = df.copy()

# print(data.sample(20))

#Discovery about the data @@@@@@@
# print(data.info())

#To check to see if any data is null
# print(data.isnull().sum())

##summary statistics of quantitative variables
# data.describe()


# as we can see below there are a few outliers so we can replace them with the median, (replace nulls with medians)
# sns.boxplot(x=data['Horsepower'])
# plt.show()


##imputing the values with median
median = data['Horsepower'].median()
data['Horsepower'] = data['Horsepower'].fillna(median)
# sns.boxplot(x=data['Horsepower'])

#information about the structure of the dataset
#data.info()
# print(data.isnull().sum())
# plt.show()



##category distribution
# print(data["Cylinders"].value_counts() / len(data))
# print(data['Origin'].value_counts())

##pairplots to get an intuition of potential correlations
sns.pairplot(data[["MPG", "Cylinders", "Displacement", "Weight", "Horsepower"]], diag_kind="kde")
# plt.show()


##below we are trying to split the dataset as the test and train groups
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["Cylinders"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

##checking for cylinder category distribution in training set and the testinge set
# print(strat_train_set['Cylinders'].value_counts() / len(strat_train_set))
# print(strat_test_set['Cylinders'].value_counts() / len(strat_test_set))

##converting integer classes to countries in Origin 
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

##converting integer classes to countries in Origin column
train_set['Origin'] = train_set['Origin'].map({1: 'India', 2: 'USA', 3 : 'Germany'})
# print(train_set.sample(10))

##one hot encoding
train_set = pd.get_dummies(train_set, prefix='', prefix_sep='')
print(train_set.head())

data = strat_train_set.copy()
