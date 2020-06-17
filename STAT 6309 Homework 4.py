#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 18:31:42 2020

@author: andreastsoumpariotis
"""

import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.formula.api as Logit
from statsmodels.api import add_constant
from statsmodels.graphics.gofplots import ProbPlot
from sklearn.metrics import confusion_matrix
import statistics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from IPython.display import display, HTML
from functools import partial 

# Question 6

dataframe = pd.read_csv('Default.csv', true_values=['Yes'], false_values=['No'])
dataframe.head(10)

default = dataframe['default']
student = dataframe['student']
balance = dataframe['balance']
income = dataframe['income']

# Part a
predictors = ['balance','income']

x = sm.add_constant(dataframe[predictors])
y = dataframe.default.values

# Logistic Regression Model
model = sm.Logit(y, x)
logistic = model.fit()
# Summary
print(logistic.summary())
print(logistic.bse) #Shows Standard Errors

# Part b

def default_boot(df, indices):
    predictors = ['balance', 'income']
    response = ['default']
    
    predictors = ['balance', 'income']
    response = ['default']
    
    x = sm.add_constant(df[predictors]).loc[indices]
    y = df[response].loc[indices]

    results = sm.Logit(y, x).fit(disp=0)
    
    return [results.params[predictors].balance, results.params[predictors].income]

# print coefficient estimates
np.random.seed(0)
indices = np.random.choice(dataframe.index, size=len(df), replace=True)
print(default_boot(dataframe, indices))

# Part c
n = 1000
def boot(data, stat_func, samples = n):
    
    BootSamples = []
    
    for sample in range(samples):
        indices = np.random.choice(data.index, size = len(data), replace = True)

        BootSamples.append(stat_func(data, indices))
    
    # compute the se estimate    
    standard_error = scipy.std(BootSamples, axis = 0)
    
    return standard_error

np.random.seed(0)
print(boot(dataframe, default_boot, n))


# Question 7

weekly = pd.read_csv('weekly.csv', true_values=['Up'], false_values=['Down'])

# Part a

x = sm.add_constant(weekly[['Lag1', 'Lag2']])
y = weekly.Direction

# Fitting the Model
model = sm.Logit(y, x)
result = model.fit()
print(result.summary())

# Part b

x = sm.add_constant(weekly[['Lag1', 'Lag2']]).loc[1:]
y = weekly.Direction.loc[1:]

# Fitting the Model
model = sm.Logit(y, x)
result = model.fit()
print(result.summary())

# Part c
x = sm.add_constant(weekly[['Lag1', 'Lag2']])
y_predicted = model.predict(x.loc[0])
print(y_predicted > 0.5)

# Part d

response = "Direction_Up"
predictors = ["Lag1", "Lag2"]

y_pred = []
for i in range(weekly.shape[0]):

    train = weekly.index != i
    
    x_training = np.array(weekly[train][predictors])
    x_test  = np.array(weekly[~train][predictors])
    y_training = np.array(weekly[train][response])
    
    logit = LogisticRegression()
    model = logit.fit(x_training, y_training)
    
    y_pred += [model.predict(x_test)]
    
y_pred = np.array(y_pred)

# part e

confusion = confusion_matrix(y_test, y_pred)
display(confusion)

def total_error_rate(confusion_matrix):

    return 1 - np.trace(confusion) / np.sum(confusion)

error = np.around(total_error_rate(confusion) * 100, 4)
print(error) #44.9954

# Question 9

boston = pd.read_csv("Boston.csv", index_col = 0)
boston.head()

# Part a

mu = boston.medv.mean()
print(mu)

# Part b

SE = boston.medv.std()/np.sqrt(len(boston))
print(SE)

# Part c

# Bootstrap Function
def boot(boston, column, statistic, samples=1000):
    
    def bootstrap(boston, column, statistic):
        
        indices = np.random.choice(boston.index, size = len(boston), replace = True)

        stat = statistic(boston[column].loc[indices])
    
        return stat
    
    return scipy.std([bootstrap(boston, column, statistic) for sample in range(samples)], axis=0)

print(boot(boston,'medv', np.mean))
# The standard error for mu that we obtain using the bootstrap: 0.4187775840605994

# Part d

# Need to obtain t-value first
t = stats.t.isf(.05/2,len(boston)-1)
print(t) #1.964672638739595

# Confidence Interval
print(np.array([mu-t*SE, mu+t*SE]))
# CI: [21.72952801, 23.33608463]

# Part e
med_hat = boston.medv.median()
med_hat #21.2

# Part f
se_med_hat = boot(boston, 'medv', np.median)
se_med_hat #0.3730549423342354

# Part g
tenth_perc_med = boston.medv.quantile(q=0.1)
tenth_perc_med #12.75

# Part h
q = partial(np.percentile, q=10)
se_tenth_perc_med = boot(boston, 'medv', q)
se_tenth_perc_med #0.5273287623484992


