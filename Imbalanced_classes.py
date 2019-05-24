# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:25:14 2019

Nate Jermain
Random forest analysis for stock delineation for Chub Mackerel
"""

import pandas as pd
import numpy as np

df=pd.read_csv("C:/Users/w10007346/Dropbox/ATC shape analsis project (1)/ACM_ShapeAnalysis/Analysis/wave_shape.csv")

df.head()
orig_resp=df['pop']

####### Data Cleaning ###########

df.columns.values
len(df.columns.values)

#### Transforming skewed features
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
df.skew().sort_values(ascending=False)# improved

##### Remove duplicate features and NAs
# remove id column
df=df.drop('Unnamed: 0', axis=1)
# response is population
resp=df['pop']
df=df.drop('pop', axis=1)

df.columns.values


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
resp=resp[0]

# Split into training and test sets
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(df, orig_resp, random_state = 0, test_size=.3)


# make sure test set is balanced
from imblearn.under_sampling import RandomUnderSampler

untest = RandomUnderSampler(return_indices=True)
X_untest, y_untest, id_untest = untest.fit_sample(test_X, test_y)

pd.Series(y_untest).value_counts().plot('bar') # equal sampling now (check)

# prior to resampling
from sklearn.ensemble import RandomForestClassifier

# the model prior to hyperparameter optimization
Best_RFC=RandomForestClassifier(n_estimators=4000, max_features='auto', max_depth=20,
                           min_samples_split=5, min_samples_leaf=1,
                           bootstrap=True, criterion='gini')

Best_RFC.fit(train_X, train_y)

# predict test Y values
ypred=Best_RFC.predict(X_untest)

# apply to test set
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_untest, ypred))

# confusion matrix to evaluate predictions
pd.crosstab(y_untest, ypred, rownames=['Observed'], colnames=['Predicted'])


########### Undersampling of common classes 
import matplotlib.pyplot as plt
fig = plt.figure()
plot= pd.Series(train_y).value_counts().plot('bar', color=['green', 'blue', 'red']) # unbalanced design
fig = plot.get_figure()
fig.savefig('Imbalanced.png', dpi=300) 


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(return_indices=True)
X_rus, y_rus, id_rus = rus.fit_sample(train_X, train_y)

pd.Series(y_rus).value_counts().plot('bar') # equal sampling now (check)

fig = plt.figure()
plot = pd.Series(y_rus).value_counts().plot('bar', color=['green', 'blue', 'red'])  
fig = plot.get_figure()
fig.savefig('oversamp.png', dpi=300) 

### MODEL

Best_RFC.fit(X_rus, y_rus)

# predict test Y values
ypred=Best_RFC.predict(X_untest)

# apply to test set
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_untest, ypred))

# confusion matrix to evaluate predictions
pd.crosstab(y_untest, ypred, rownames=['Observed'], colnames=['Predicted'])




######## Oversampling of rare classes  ######################
# lets try oversampling too
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(train_X, train_y)

# equal sampling now (check)
fig = plt.figure()
plot = pd.Series(y_ros).value_counts().plot('bar', color=['green', 'blue', 'red'])  
fig = plot.get_figure()
fig.savefig('oversamp.png', dpi=300) 

# fit best model to training dataset
Best_RFC.fit(X_ros, y_ros)

# predict test Y values
ypred=Best_RFC.predict(X_untest)

# apply to test set
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_untest, ypred))

# confusion matrix to evaluate predictions
pd.crosstab(y_untest, ypred, rownames=['Observed'], colnames=['Predicted'])


########## Using SMOTE ##############
from imblearn.over_sampling import SMOTE
smot = SMOTE()
X_smot, y_smot = smot.fit_sample(train_X, train_y)


pd.Series(y_smot).value_counts().plot('bar') # equal sampling now (check)


# fit best model to training dataset
Best_RFC.fit(X_smot, y_smot)

# predict test Y values
ypred=Best_RFC.predict(X_untest)

# apply to test set
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_untest, ypred))

# confusion matrix to evaluate predictions
pd.crosstab(y_untest, ypred, rownames=['Observed'], colnames=['Predicted'])








