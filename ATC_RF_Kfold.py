# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:25:14 2019

Nate Jermain
Random forest analysis for stock delineation for Chub Mackerel
"""

import pandas as pd
import numpy as np
import seaborn as sns

df=pd.read_csv("C:/Users/w10007346/Dropbox/ATC shape analsis project (1)/ACM_ShapeAnalysis/Analysis/wave_shape.csv")

df.head()

####### Data Cleaning ###########


df.columns.values
len(df.columns.values)

#### transforming skewed features
# check features for skew
skew_feats=df.drop('pop', axis=1).skew().sort_values(ascending=False)
skewness=pd.DataFrame({'Skew':skew_feats})
skewness=skewness[abs(skewness)>0.75].dropna()

# use box cox transformation
from scipy.special import boxcox1p
skewed_features=skewness.index
lam=0.15

for i in skewed_features:
    df[i]=boxcox1p(df[i],lam)

# check
df.skew().sort_values(ascending=False)
# improved

# remove id column
df=df.drop('Unnamed: 0', axis=1)
# response is population
resp=df['pop']
df=df.drop('pop', axis=1)

df.columns.values

# plot response variable
resp.value_counts().plot('bar')


# fill in missing values
df.isnull().sum().sort_values(ascending=False)

# remove features with all nas
df=df.drop(['Ws1c1','Ws1c2', 'Ws2c4'], axis=1)

df.isnull().sum().sort_values(ascending=False)

# features with small numbers of nas get filled with means
df.Ws2c3.describe()
df.Ws2c3=df.Ws2c3.fillna(df.Ws2c3.dropna().mean())

df.Ws2c1.describe()
df.Ws2c1=df.Ws2c1.fillna(df.Ws2c1.dropna().mean())

df.isnull().sum().sort_values(ascending=False)
df.Ws3c1.describe()
df.Ws3c1=df.Ws3c1.fillna(df.Ws3c1.dropna().mean())
df.Ws4c10=df.Ws4c10.fillna(df.Ws4c10.dropna().mean())
df.Ws3c4=df.Ws3c4.fillna(df.Ws3c4.dropna().mean())
df.isnull().values.any()

# create response variable as factor
resp=pd.factorize(resp)
len(resp[0])
resp=resp[0]

# Split into training and test sets
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(df, resp, random_state = 0, test_size=.2)



############# Modeling #################
from sklearn.ensemble import RandomForestClassifier

# the model prior to hyperparameter optimization
RFC=RandomForestClassifier(n_estimators=4000)

#### use grid search to identify the best hyperparameters using Kfold CV
# max number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# max number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# quality of each split
criterion= ['gini', 'entropy']
# grid to feed gridsearch
grid_param = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion':criterion }

# gridsearch 
from sklearn.model_selection import GridSearchCV
gd_sr = GridSearchCV(estimator=RFC, param_grid=grid_param, scoring='accuracy', cv=5,n_jobs=-1)

gd_sr.fit(train_X, train_y)  
print(gd_sr.best_params_)
# {'bootstrap': False, 'criterion': 'gini', 'max_depth': 20, 
# 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5}

# update model with optimal hyperparameter values
Best_RFC=RandomForestClassifier(n_estimators=8000, max_features='auto', max_depth=20,
                           min_samples_split=5, min_samples_leaf=1,
                           bootstrap=False, criterion='gini')

# fit best model to training dataset
Best_RFC.fit(train_X, train_y)

# predict test Y values
ypred=Best_RFC.predict(test_X)

# apply to test set

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(test_y, ypred))


# confusion matrix to evaluate predictions
pd.crosstab(test_y, ypred, rownames=['Observed'], colnames=['Predicted'])














