#%%
import pandas as pd
import numpy as np

#Reading dataset

dataset = pd.read_csv('Data.csv')
print(dataset)

#Splitting test and train data
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
print(X)
print(Y)

#Taking care of the missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)

#Encoding data categorically
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = ct.fit_transform(X)
print(X)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
Y = lb.fit_transform(Y)
print(Y)

#Splitting data into test and train
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, test_size=0.2, random_state=1)
print(X_Train)

# %%
