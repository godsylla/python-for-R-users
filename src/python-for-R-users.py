
### -------- IMPORTING PACKAGES -------- ###
import numpy as np
import pandas as pd
from zipfile import ZipFile

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


### -------- TOY DATASETS -------- ### 
from sklearn.datasets import load_boston
# Load Data
boston = load_boston()
# Convert the data to a pandas DataFrame
boston_df = pd.DataFrame(boston.data)
# Load in the Target
boston_target = boston.target


### -------- LOAD DATA (USING ZIPFILE & PANDAS) -------- ### 
with ZipFile('../data/craft-cans.zip', 'r') as zipObject:
  zipObject.extractall('../data/')
  
beers_df = pd.DataFrame('../data/beers.csv')
breweries_df = pd.DataFrame('../data/breweries.csv')


### -------- CURSORY DATA INSPECTION ON THE PANDAS DATAFRAME -------- ### 
print(beers_df.info()) # for null val count & data types per column
print(beers_df.head()) # for first 5 rows
print(beers_df.describe()) # for summary stats

# Drop the index column that was read in as 'Unnamed: 0'
beers_df.drop('Unnamed: 0', axis=1, inplace=True)

# Summary value counts, defaults to return in descending order
beers_df['style'].value_counts()[:10]  # limits to top 10


### -------- LIGHT DATA CLEANING -------- ### 
# Impute null values
beers_df['abv'] = beers_df['abv'].fillna(beers_df['abv'].median())
beers_df['ibu'] = beers_df['ibu'].fillna(beers_df['ibu'].median())

# Drop records with null values
beers_df = beers_df.dropna()


### -------- SEPARATE TARGET FROM FEATURES -------- ### 
dummies_df = pd.get_dummies(beers_df.drop(['id', 'name', 'brewery_id'], axis=1), 
                            columns=['style'])
display(dummies_df.head())
X = dummies_df.drop('abv', axis=1)

y = dummies_df['abv']


### -------- TRAIN TEST SPLIT THE DATA -------- ###
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

print('X train shape: ', X_train.shape)
print('y train shape: ', y_train.shape)
print(' ')
print('X test shape: ', X_test.shape)
print('y test shape: ', y_test.shape)


### -------- FIT THE MODEL -------- ###

# Create a pipeline, which will execute in order
# the pipeline built scales the data first, 
# and finishes with the estimator as the last step
rfr_pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('rfreg', RandomForestRegressor(random_state=42, verbose=0))
])

# Implement cross-validation using GridSearchCV
# Create a dictionary object with the different parameters you want to try
grid_params = {
    'rfreg__n_estimators':[10, 15, 20]
}

# Use GridSearchCV to search different hyperparameters
# params: the pipeline (order by which you want things to run)
# params: the hyperparamters you want for it to try
# params: cv (# of cross-validations you would like to have)
# n_jobs = -1 (# parallel compute to utilize all CPU cores)
my_gridsearch = GridSearchCV(rfr_pipe, grid_params, verbose=0, cv=5, n_jobs=-1)

# Fit the model
gs_obj = my_gridsearch.fit(X_train, y_train)
print(gs_obj)

# Of all the models attempted, return the best estimator, its best parameters, and the best score
print('best estimator: ' , gs_obj.best_estimator_, ' ')
print('best parameters: ', gs_obj.best_params_, ' ')
print('best score: ', gs_obj.best_score_)

# How to access the methods & attributes of the estimator:
# This returns a dictionary object
print(gs_obj.best_estimator_.named_steps)

# Access the dictionary value by using the key: 'rfreg'
print(gs_obj.best_estimator_.named_steps['rfreg'])
 

### -------- MODEL RESULTS -------- ### 

# Inspect all cross-validated results and convert to a DataFrame 
cv_results = pd.DataFrame(gs_obj.cv_results_).T
print(cv_results)

# Get feature importances
feat_imptc = gs_obj.best_estimator_.named_steps['rfreg'].feature_importances_

# Zip w/ column names & sort, display top 10
sorted(list(zip(feat_imptc, X.columns)), reverse=True)[:10]

# MSE for test (validation) data
test_score = gs_obj.best_estimator_.score(X_test, y_test)
print(test_score)

