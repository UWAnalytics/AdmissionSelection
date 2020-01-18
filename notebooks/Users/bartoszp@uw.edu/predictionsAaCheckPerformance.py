# Databricks notebook source
# MAGIC %md # AdSel Modeling
# MAGIC ## Step 3: Modeling
# MAGIC ### Step 3b: Check/validate model performance
# MAGIC 
# MAGIC The purpose of this script is to test the performance of the AA predictive models. This is an optional script in the pipeline.
# MAGIC 
# MAGIC ### Goals
# MAGIC 
# MAGIC * Determine performance of AA predictive models
# MAGIC 
# MAGIC ### Process
# MAGIC 
# MAGIC * A. Load data and modules
# MAGIC * B. Set configurations
# MAGIC * C. Preprocessing
# MAGIC * D. Splits and scaling
# MAGIC * E. Predictions
# MAGIC * F. Check predictions
# MAGIC * G. Check prediction performance
# MAGIC * H. Save errors
# MAGIC * I. Historic predictions

# COMMAND ----------

# MAGIC %md ## Part A - Load data and modules

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

# COMMAND ----------

file_location = "/dbfs/FileStore/tables/dataForAaPredictions.csv"
data = pd.read_csv(file_location)

# COMMAND ----------

data.head()

# COMMAND ----------

data.columns

# COMMAND ----------

# MAGIC %md ## Part B - Set configurations

# COMMAND ----------

saveErrors = True

# COMMAND ----------

allData = data.copy()
allData = allData[allData.entryYear < 2020]

# COMMAND ----------

studentTag = 'non-resident' #what types of students to look at? resident, non-resident, international

allData = allData[allData[studentTag] == 1]
paramLocDict = {
    'resident': '../../outputs/resAaParams.pkl',
    'non-resident' : '../../outputs/dnrAaParams.pkl',
    'international' : '../../outputs/intlAaParams.pkl'
}
paramLoc = paramLocDict[studentTag]

# COMMAND ----------

if os.path.isfile(paramLoc):
    with open(paramLoc, 'rb') as pickleIn:
        params = pickle.load(pickleIn)
else:
    params = {}

# COMMAND ----------

params

# COMMAND ----------

len(allData)

# COMMAND ----------

# MAGIC %md ## Part C - Preprocessing

# COMMAND ----------

ax = plt.subplot(111)
p = sns.distplot(allData.aaScore, bins = np.arange(0, allData.aaScore.max() + 2), norm_hist = False, 
                 kde = False)
p.set(xticks= np.arange(0.5, allData.aaScore.max() + 1.5), 
      xticklabels = [str(int(x)) for x in np.arange(0, allData.aaScore.max() + 1)],
      zorder = 10);

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.grid(axis = 'y', zorder = 1)

plt.xlim([0, 19])

plt.xlabel('Academic Sub Score')
plt.ylabel('Count')

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

allData = allData[allData.aaScore.notna()]

# COMMAND ----------

# MAGIC %md ## Part D - Splits and scaling

# COMMAND ----------

# MAGIC %md ### Train/test split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(allData.drop(['aaScore'], axis = 1), 
                                                    allData.aaScore, test_size = 0.2, random_state = 11)

# COMMAND ----------

y_test = pd.DataFrame(y_test)
y_test['hold'] = 1
y_train = pd.DataFrame(y_train)
y_train['hold'] = 1

# COMMAND ----------

xcols = X_train.columns
ycols = y_train.columns

# COMMAND ----------

# MAGIC %md ### Scaling

# COMMAND ----------

scaler = MinMaxScaler().fit(X_train)

# COMMAND ----------

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# COMMAND ----------

scalerY = MinMaxScaler().fit(y_train)

# COMMAND ----------

y_train = scalerY.transform(y_train)
y_test = scalerY.transform(y_test)

# COMMAND ----------

X_train = pd.DataFrame(X_train, columns = xcols)
X_test = pd.DataFrame(X_test, columns = xcols)
y_train = pd.DataFrame(y_train, columns = ycols)
y_test = pd.DataFrame(y_test, columns = ycols)

# COMMAND ----------

y_train = y_train.drop(['hold'], axis = 1)
y_test = y_test.drop(['hold'], axis = 1)

# COMMAND ----------

y_train.hist()

# COMMAND ----------

# MAGIC %md ## Part E - Predictions

# COMMAND ----------

# MAGIC %md ### "Actual" predictions

# COMMAND ----------

xgb = GradientBoostingRegressor(**params)
xgb.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md ### Lower bound

# COMMAND ----------

xgbLow = GradientBoostingRegressor(**params, loss='quantile', alpha = 0.05)
xgbLow.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md ### Upper bound

# COMMAND ----------

xgbHigh = GradientBoostingRegressor(**params, loss='quantile', alpha = 0.95)
xgbHigh.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md ## Part F - Check predictions

# COMMAND ----------

# MAGIC %md ### Density plots

# COMMAND ----------

preds = pd.DataFrame(xgb.predict(X_train))
preds['hold'] = 0
y_rescaled = y_train.copy()
y_rescaled['hold'] = 0

# COMMAND ----------

preds_rescaled = scalerY.inverse_transform(preds)
y_rescaled = scalerY.inverse_transform(y_rescaled)

# COMMAND ----------

predsAll = pd.DataFrame(xgb.predict(X_train.append(X_test)))
predsAll['hold'] = 0
preds_all_rescaled = scalerY.inverse_transform(predsAll)[:,0]

# COMMAND ----------

ax = plt.subplot(111)
p = sns.distplot(allData.aaScore, bins = np.arange(0, allData.aaScore.max() + 2))
p.set(xticks= np.arange(0.5, allData.aaScore.max() + 1.5), 
      xticklabels = [str(int(x)) for x in np.arange(0, allData.aaScore.max() + 1)],
      zorder = 10);

p2 = sns.distplot(preds_all_rescaled, bins = np.arange(0, allData.aaScore.max() + 2), color = 'red')
p2.set(xticks= np.arange(0.5, allData.aaScore.max() + 1.5), 
      xticklabels = [str(int(x)) for x in np.arange(0, allData.aaScore.max() + 1)],
      zorder = 10);

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.grid(axis = 'y', zorder = 1)

plt.xlim([0, 19])

plt.legend(['Actual', 'Predicted'])

plt.xlabel('Academic Sub Score')
plt.ylabel('Density')

# COMMAND ----------

# MAGIC %md ### Errors

# COMMAND ----------

errors = (preds_rescaled[:,0] - y_rescaled[:,0])

# COMMAND ----------

sns.distplot(errors)

# COMMAND ----------

sns.boxplot(errors)

# COMMAND ----------

plt.scatter(preds_rescaled, y_rescaled)
plt.plot(np.arange(0, 19), np.arange(0, 19), 'r-')

# COMMAND ----------

# MAGIC %md ### Create error DF

# COMMAND ----------

preds_test_xgb = pd.DataFrame(xgb.predict(X_test))
preds_test_xgb['hold'] = 0

# COMMAND ----------

preds_test_xgb_low = pd.DataFrame(xgbLow.predict(X_test))
preds_test_xgb_low['hold'] = 0

# COMMAND ----------

preds_test_xgb_high = pd.DataFrame(xgbHigh.predict(X_test))
preds_test_xgb_high['hold'] = 0

# COMMAND ----------

y_test_rescaled = y_test.copy()
y_test_rescaled['hold'] = 0

# COMMAND ----------

preds_test_xgb = scalerY.inverse_transform(preds_test_xgb)[:,0]
preds_test_xgb_low = scalerY.inverse_transform(preds_test_xgb_low)[:,0]
preds_test_xgb_high = scalerY.inverse_transform(preds_test_xgb_high)[:,0]
y_test_rescaled = scalerY.inverse_transform(y_test_rescaled)[:,0]

# COMMAND ----------

outs = pd.DataFrame({'low': preds_test_xgb_low, 'mid': preds_test_xgb, 'high': preds_test_xgb_high, 'actual': y_test_rescaled})

# COMMAND ----------

outs['check'] = (outs['actual'] >= outs['low']) & (outs['actual'] <= outs['high'])

# COMMAND ----------

outs

# COMMAND ----------

outs.check.sum() / len(outs) #How many times are we between the bounds established?

# COMMAND ----------

# MAGIC %md ### Round predictions

# COMMAND ----------

plt.scatter(preds_test_xgb, y_test_rescaled)
plt.plot(np.arange(0, 19), np.arange(0, 19), 'r-')

# COMMAND ----------

np.sqrt(mean_squared_error(preds_test_xgb, y_test_rescaled))

# COMMAND ----------

rounded = np.round(preds_test_xgb)

# COMMAND ----------

plt.scatter(rounded, y_test_rescaled, alpha = 0.1)

# COMMAND ----------

yhatSeries = pd.Series(rounded)

# COMMAND ----------

ySeries = pd.Series(y_test_rescaled)

# COMMAND ----------

plt.hist(rounded, bins = 14)

# COMMAND ----------

plt.hist(y_test_rescaled, bins = 16)

# COMMAND ----------

diffs = ySeries - rounded

# COMMAND ----------

plt.hist(diffs)

# COMMAND ----------

# MAGIC %md ## Part G - Check prediction performance

# COMMAND ----------

counts = np.round(np.abs(diffs)).value_counts()
counts

# COMMAND ----------

# Bar errors: what pct of time are we correct? Off by <=1 point? Off by <=2 points?
total = counts.sum()

bars = [counts[0] / total * 100, (counts[0] + counts[1]) / total * 100, 
 (counts[0] + counts[1] + counts[2]) / total * 100]
bars

# COMMAND ----------

# Similar to above only not using rounded values
total = len(errors)

abs_errors = np.abs(errors)
cumSteps = []
cum = []
step = 0.001
for current in np.arange(0, 2.5 + step, step):
    value = len(abs_errors[abs_errors < current])
    cum.append(value)
    if current in [0.1, 0.5, 1, 2]:
        print (current, value, value / total * 100)
        cumSteps.append((current, value / total * 100))
    
cum = np.array(cum) / total * 100

# COMMAND ----------

#plot of bars from above
ax = plt.subplot(111)
plt.plot(np.arange(0, 2.5 + step, step), cum)
plt.bar([0, 1, 2], bars, width = 0.2, alpha = 0.5)
plt.plot([])
plt.xlim([-0.15, 2.5])
plt.ylim(-1, 101)


# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.grid(axis = 'y')

plt.xlabel('Error')
plt.ylabel('Pct of Held Out (cumulative)')

# COMMAND ----------

print ("RMSE: ", np.sqrt(mean_squared_error(preds_test_xgb, y_test_rescaled)))
print ("Median: ", np.median(np.abs(preds_test_xgb - y_test_rescaled)))
print ("Bar errors:", bars)
print ("Cum errors:", cumSteps)

# COMMAND ----------

# MAGIC %md ## Part H - Save errors

# COMMAND ----------

if saveErrors:
    errorFile = '/dbfs/FileStore/tables/errorsAa' + studentTag.title().replace('-','') + '.csv'
    outs.to_csv(errorFile, index = False)

# COMMAND ----------

# MAGIC %md ## Part I - Historic predictions

# COMMAND ----------

#RES

#RMSE:  0.9981273978337988
#Median:  0.6192058332745018
#Bar errors: [41.260245901639344, 87.73565573770492, 98.20696721311475]
#Cum errors: [(0.1, 9.88318475253612), (0.5, 46.677425965775186), (1.0, 77.66676913618198), (2.0, 97.589404652116)]

# COMMAND ----------

#DNR

#RMSE:  1.0813340900226154
#Median:  0.6629645098934969
#Bar errors: [38.99278121980333, 85.19865855738078, 97.16364463138748]
#Cum errors: [(0.1, 10.444791814693762), (0.5, 49.10046894983658), (1.0, 79.87636777035668), (2.0, 97.85419923262754)]

# COMMAND ----------

#INTL

#RMSE:  2.329473159820949
#Median:  0.9684474713059008
#Bar errors: [33.27356557377049, 63.70389344262295, 79.41854508196722]
#Cum errors: [(0.1, 11.142418032786885), (0.5, 41.26857069672131), (1.0, 62.247054303278695), (2.0, 82.95658299180327)]