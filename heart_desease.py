import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import os
import warnings

warnings.filterwarnings('ignore')

# print(os.listdir("../input"))
PATH = os.getcwd() + '\\input'
print(PATH)

data = pd.read_csv(PATH + '\heart.csv')

# Now, our data is loaded. We're writing the following snippet to see the loaded data.
# The purpose here is to see the top five of the loaded data.
print('Data First 5 Rows Show\n')
print(data.head())

print('Data Last 5 Rows Show\n')
print(data.tail())

print('Data Show Describe\n')
print(data.describe())

print('Data Show Info\n')
print(data.info())

# We will list all the columns for all data. We check all columns. Is there any spelling mistake?
print('Data Show Columns:\n')
print(data.columns)

print(data.sample(frac=0.01))

# sample; random rows in dataset
print(data.sample(5))

data = data.rename(
    columns={'age': 'Age', 'sex': 'Sex', 'cp': 'Cp', 'trestbps': 'Trestbps', 'chol': 'Chol', 'fbs': 'Fbs',
             'restecg': 'Restecg', 'thalach': 'Thalach', 'exang': 'Exang', 'oldpeak': 'Oldpeak', 'slope': 'Slope',
             'ca': 'Ca', 'thal': 'Thal', 'target': 'Target'})

# New show columns
print(data.columns)

# And, how many rows and columns are there for all data?
print('Data Shape Show\n')
print(data.shape)  # first one is rows, other is columns

# Now,I will check null on all data and If data has null, I will sum of null data's.
# In this way, how many missing data is in the data.
print('Data Sum of Null Values \n')
print(data.isnull().sum())

# all rows control for null values
print('All rows control for null values \n')
print(data.isnull().values.any())

plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, fmt='.1f')
plt.show()

plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.tight_layout()
plt.show()

sns.pairplot(data)
plt.show()

# data age show value counts for age least 10
print('Data age show value counts for age least 10 \n')
print(data.Age.value_counts()[:10])

sns.barplot(x=data.Age.value_counts()[:10].index, y=data.Age.value_counts()[:10].values)
plt.xlabel('Age')
plt.ylabel('Age Counter')
plt.title('Age Analysis System')
plt.show()

# firstly find min and max ages
minAge = min(data.Age)
maxAge = max(data.Age)
meanAge = data.Age.mean()
print('Min Age :', minAge)
print('Max Age :', maxAge)
print('Mean Age :', meanAge)

young_ages = data[(data.Age >= 29) & (data.Age < 40)]
middle_ages = data[(data.Age >= 40) & (data.Age < 55)]
elderly_ages = data[(data.Age > 55)]
print('Young Ages :', len(young_ages))
print('Middle Ages :', len(middle_ages))
print('Elderly Ages :', len(elderly_ages))

sns.barplot(x=['young ages', 'middle ages', 'elderly ages'], y=[len(young_ages), len(middle_ages), len(elderly_ages)])
plt.xlabel('Age Range')
plt.ylabel('Age Counts')
plt.title('Ages State in Dataset')
plt.show()

# MODEL, TRAINING and TESTING

# Let's see how the correlation values between them
print(data.corr())
