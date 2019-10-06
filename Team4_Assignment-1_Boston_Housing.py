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


# In[2]:


# Define Download Path – Note 1st row of csv is not part of data so skiprows=1 when reading csv
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/master/sklearn/"
HOUSING_PATH = os.path.join(DOWNLOAD_ROOT,"datasets","data")
HOUSING_URL = "boston_house_prices.csv"

def load_housing_data(housing_path=HOUSING_PATH,housing_url=HOUSING_URL):
    csv_path = os.path.join(housing_path, housing_url)
    return pd.read_csv(csv_path, skiprows = 1)

housing = load_housing_data()
housing.head()


# In[3]:


housing.info()


# In[4]:


housing.describe()


# In[5]:


# Look for any null inputs in the data
housing.isnull().sum()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[7]:


# to make this notebook's output identical at every run
np.random.seed(42)


# In[8]:


# Split training and test sets
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(len(train_set), "train +", len(test_set), "test")


# In[9]:


# Find Correlations in the training sets
housing = train_set.copy()
corr_matrix = housing.corr()
corr_matrix["MEDV"].sort_values(ascending=False)


# In[10]:


# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ["MEDV", "RM", "LSTAT"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[11]:


# Standardize attributes to prepare for ML models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

housing = train_set.drop("MEDV",axis=1)
num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])

housing_prepared = num_pipeline.fit_transform(housing)
housing_labels = train_set["MEDV"].copy()
housing_prepared


# In[12]:


# Function to print scores for each model
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[13]:


# Train Linear Regression Model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Cross-validation
from sklearn.model_selection import cross_val_score
lin_scores = cross_val_score (lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[14]:


# Train Decision Tree Regression Model
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

# Cross-validation
tree_scores = cross_val_score (tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)


# In[15]:


# Train Random Forest Regressor Model
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

# Cross-validation
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[16]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

final_model = grid_search.best_estimator_
X_test = test_set.drop("MEDV", axis=1)
Y_test = test_set["MEDV"].copy()
X_test_prepared = num_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[ ]:




