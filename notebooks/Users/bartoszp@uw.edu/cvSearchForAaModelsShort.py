# Databricks notebook source
# MAGIC %md # AdSel Modeling
# MAGIC ## Step 3: Modeling
# MAGIC ### Step 3a: Cross validate model hyperparameters
# MAGIC 
# MAGIC The purpose of this script is to optimal hyperparameters for the AA predictive models. This is an optional script in the pipeline. Note - this will take a (very) long time to run.
# MAGIC 
# MAGIC ### Goals
# MAGIC 
# MAGIC * Use cross validation to find optimal hyperparameters for AA predictive models
# MAGIC 
# MAGIC ### Process
# MAGIC 
# MAGIC * A. Load data and modules
# MAGIC * B. Set configurations
# MAGIC * C. Preprocessing
# MAGIC * D. Cross-validation

# COMMAND ----------

# MAGIC %md ## Part A - Load data and modules

# COMMAND ----------

import pandas as pd
import pickle
import numpy as np
import multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

# COMMAND ----------

data = pd.read_csv('/dbfs/FileStore/tables/dataForAaPredictions.csv')

# COMMAND ----------

data.head()

# COMMAND ----------

data.columns

# COMMAND ----------

# MAGIC %md ## Part B - Set configurations

# COMMAND ----------

save_results = True

# COMMAND ----------

cores = multiprocessing.cpu_count()

# COMMAND ----------

parameters = { #these can be altered/fine-tuned
    'max_depth': [5, 7, 9],
    'learning_rate': [1, 0.1, 0.01],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [0.5, 0.75, 0.9, 1.0]
}

# COMMAND ----------

# MAGIC %md ## Part C - Preprocessing

# COMMAND ----------

allData = data.copy()
allData = allData[allData.aaScore.notna()]

# COMMAND ----------

len(allData)

# COMMAND ----------

cols = []
for col in allData.columns:
    current = allData[col]
    for each in current:
        try:
            float(each)
        except ValueError:
            cols.append(col)

# COMMAND ----------

cols = list(set(cols))

# COMMAND ----------

cols

# COMMAND ----------

allData = allData.drop(cols, axis = 1).astype(float)

# COMMAND ----------

allData.entryYear.value_counts()

# COMMAND ----------

# MAGIC %md ## Part D - Cross-validation

# COMMAND ----------

# MAGIC %md ### First, resident students

# COMMAND ----------

currentData = allData[allData['resident'] == 1].copy()

# COMMAND ----------

currentX = currentData.drop('aaScore', axis = 1)
currentY = currentData[['aaScore']]

# COMMAND ----------

scalerResX = MinMaxScaler()
scaledX = scalerResX.fit_transform(currentX)
scalerResY = MinMaxScaler()
scaledY = scalerResY.fit_transform(currentY)

# COMMAND ----------

xgbRes = GradientBoostingRegressor()

# COMMAND ----------

clf = GridSearchCV(xgbRes, parameters, n_jobs = cores - 1, verbose = 1)

# COMMAND ----------

clf.fit(scaledX, scaledY)

# COMMAND ----------

paramsRes = clf.best_params_
paramsRes

# COMMAND ----------

if save_results: #save prediction model
    output_name = '/dbfs/FileStore/resAaParams_parallel.pkl'
    with open(output_name, 'wb') as output:
        pickle.dump(paramsRes, output, pickle.HIGHEST_PROTOCOL)