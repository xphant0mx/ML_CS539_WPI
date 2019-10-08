#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import tarfile
from six.moves import urllib
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# Load the diabetes dataset
from sklearn import datasets, linear_model
diabetes = datasets.load_diabetes()

# Format dataset into a data frame
Xcolumns = "AGE SEX BMI ABP S1 S2 S3 S4 S5 S6".split()
diabetes_X = pd.DataFrame(diabetes.data, columns=Xcolumns)

Ycolumns = "TARG".split()
diabetes_Y = pd.DataFrame(diabetes.target, columns=Ycolumns)
diabetes_ = pd.concat([diabetes_X, diabetes_Y], axis=1)

print(diabetes_X.shape)
print(diabetes_Y.shape)
print(diabetes_.shape)


# In[2]:


# Split the data into test/training sets for cross-validation
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(diabetes_X, diabetes_Y, test_size=0.20)

print(len(X_train), "+", len(Y_train), "and", len(X_test), "+", len(Y_test))


# In[3]:


# Function to print scores for each model
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[4]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error 

rmse_val = [] #to store rmse values for different K
K = 0
for K in range(30):
    K = K+1
    # K Nearest Neighbor Regressor model
    knn_diabetes = KNeighborsRegressor(n_neighbors = K)
    knn_predict = cross_val_predict(knn_diabetes, X_train, Y_train)
    #Store RMSE (pred vs data) score for each K
    error = sqrt(mean_squared_error(Y_train,knn_predict)) #calculate rmse
    rmse_val.append(error) #store rmse values
    #print('RMSE value for k= ' , K , 'is:', error)
    print('RMSE value for k= ' , K , 'is:', error)


# In[5]:


plt.plot(rmse_val)
plt.xlabel('Value of K-1 for KNN')
plt.ylabel('Cross-Validated RMSE')


# In[6]:


# Find correlations of each attribute to the target
corr_matrix = diabetes_.corr()
corr_matrix["TARG"].sort_values(ascending=False)


# In[7]:


# Visualize prediction vs data for K=15
K = 15
knn_diabetes = KNeighborsRegressor(n_neighbors=K)
knn_predict = cross_val_predict(knn_diabetes, X_train, Y_train)
BMI = X_train["BMI"].copy()
plt.scatter(BMI, Y_train, c='k', label='data')
plt.scatter(BMI, knn_predict, c='g', label='prediction')
plt.legend()
plt.xlabel('BMI')
plt.ylabel('Target')
plt.title('KNN Regressor, K=15')


# In[8]:


# Visualize prediction vs data for K=1
K = 1
knn_diabetes = KNeighborsRegressor(n_neighbors=K)
knn_predict = cross_val_predict(knn_diabetes, X_train, Y_train)
plt.scatter(BMI, Y_train, c='k', label='data')
plt.scatter(BMI, knn_predict, c='g', label='prediction')
plt.legend()
plt.xlabel('BMI')
plt.ylabel('Target')
plt.title('KNN Regressor, K=1')


# In[9]:


# Average RMSE for KNN Regressor with K =15 for Test Data
K = 15
knn_diabetes = KNeighborsRegressor(n_neighbors=K)
#Predict target values based on test data
knn_diabetes.fit(X_train, Y_train)
n15_scores = cross_val_score(knn_diabetes,  X_test, Y_test, scoring="neg_mean_squared_error")
n15_rmse = np.sqrt(-n15_scores)
display_scores(n15_rmse)


# In[10]:


from sklearn.model_selection import GridSearchCV

# define weight options: 'uniform' is equally weighted and 'distance' takes closer points more heavy
weight_options = ['uniform', 'distance']
# define the parameter values that should be searched
k_range = list(range(1, 30))
param_grid = dict(n_neighbors=k_range, weights=weight_options)
# define Regressor
knn_grid = KNeighborsRegressor()

# Train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid = GridSearchCV(knn_grid, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid.fit(X_train, Y_train)
print(grid.best_score_)
print(grid.best_params_)


# In[11]:


# Visualization using best combination: k=18 and 'distance' weight
knn_diabetes = KNeighborsRegressor(n_neighbors=18, weights = 'distance')
knn_predict = cross_val_predict(knn_diabetes, X_train, Y_train)
BMI = X_train["BMI"].copy()
plt.scatter(BMI, Y_train, c='k', label='data')
plt.scatter(BMI, knn_predict, c='g', label='prediction')
plt.legend()
plt.xlabel('BMI')
plt.ylabel('Target')
plt.title('KNN Regressor, K=18')


# In[12]:


# Average RMSE for KNN Regressor with K =18 for Test Data
#Predict target values based on test data
knn_diabetes.fit(X_train, Y_train)
n18_scores = cross_val_score(knn_diabetes,  X_test, Y_test, scoring="neg_mean_squared_error")
n18_rmse = np.sqrt(-n18_scores)
display_scores(n18_rmse)


# In[ ]:




