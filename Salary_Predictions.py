#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 07:01:07 2019

@author: raviswanath

@email: aadithya.viswanath@gmail.com

"""

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# Scikit Learn pre-processing libraries needed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# sklearn ML libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import xgboost as xgb

# New regression base lib which helps with building visuals of the results
from yellowbrick.regressor import ResidualsPlot

# load the data into a Pandas dataframe
Test_Features = pd.read_csv('/Users/raviswanath/Dropbox/data/'
                            'test_features.csv')
Train_Features = pd.read_csv('/Users/raviswanath/Dropbox/data/'
                             'train_features.csv')
Train_Salaries = pd.read_csv('/Users/raviswanath/Dropbox/data/'
                             'train_salaries.csv')

Training_data = Train_Features.merge(Train_Salaries, on="jobId")

# Function to remove records which have less than x% of
# rows in a certain salary range


def remove_extremes(Training_data, tune_param):
    '''Function to remove records which have less than x%
    of rows in a certain salary range'''

    rows_removed = 0
    Job_Types = Training_data.jobType.unique()

    for j in Job_Types:
        x = Training_data[Training_data.jobType == j]['salary'].count()
        for i in range(0, 100, 10):
            # identify records in the certain bin for jobType j
            y = Training_data[(Training_data.jobType == j) &
                (Training_data.salary >= np.percentile(Training_data.salary,
                                                       i)) &
                (Training_data.salary < np.percentile(Training_data.salary,
                                                      (i+10)))]['salary'].count()

            if((y/x)*100 < tune_param):
                    # keep count of the number of rows removed
                    rows_removed = rows_removed + y

                    # eliminate the records
                    Training_data = Training_data.loc[~((Training_data.jobType == j)&
                    (Training_data.salary >= np.percentile(Training_data.salary, i))&
                    (Training_data.salary < np.percentile(Training_data.salary, (i+10)))), :]
    print(rows_removed)
    return(Training_data)


# remove extremes
Training_data = remove_extremes(Training_data, 1)
Train_Salaries = Training_data['salary']
Training_data.drop('salary', axis=1, inplace=True)
Train_Features = Training_data

# creating a validation set to help tune to boosting model
Train_Features, Valid_Features, Train_Salaries, Valid_Salaries = train_test_split(Train_Features,
                                                                                  Train_Salaries,
                                                                                 test_size=0.3)

# Select categorical columns and check their cardinality
categorical_cols = [cols for cols in Train_Features.columns
                   if Train_Features[cols].dtype == 'object']

# only keep columns which have a cardinality < 10
categorical_cols = [col for col in categorical_cols
                   if Train_Features[col].nunique() <= 10]

# numerical cols
numeric_cols = [col for col in Train_Features.columns
               if Train_Features[col].dtype in ["int64", "float64"]]

# removing high cardinal features from training and validation sets
all_cols = numeric_cols + categorical_cols
Train_Features = Train_Features[all_cols]
Valid_Features = Valid_Features[all_cols]


# There is no missing data, however, we define imputers
# to make the code production ready

numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer_jobtype = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='SENIOR')),
    ('encdoer', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


categorical_transformer_industry = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='WEB')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

categorical_transformer_others = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='NONE')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


# Bundle the pre-processing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat1', categorical_transformer_jobtype, ['jobType']),
        ('cat2', categorical_transformer_industry, ['industry']),
        ('cat3', categorical_transformer_others, ['degree', 'major'])
    ])

# Model pipeline

model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', xgb.XGBRegressor(n_jobs=-1, n_estimators=200))
])

param = {
    'model__max_depth': [i for i in range(3, 10, 2)],
    'model__learning_rate': [3e-1, 2e-1, 1e-1],
    'model__min_child_weight': [j for j in range(5, 25, 5)],
    'model__colsample_bytree': [7e-1, 8e-1, 9e-1]
}

fit_params = {
    "model__early_stopping_rounds": 5,
    "model__eval_metric": 'rmse',
    "model__eval_set": [(preprocessor.fit_transform(Valid_Features),
                         Valid_Salaries)]
}


grid_model = RandomizedSearchCV(model_pipeline,
                          param_distributions=param, cv=3, verbose=False,
                          fit_params=fit_params).fit(Train_Features,
                                               Train_Salaries)
