#Simple Linear Regression
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

dataset = pd.read_csv('Salary_Data.csv')
print(dataset)

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=1)
print(X_Train)



#%%

