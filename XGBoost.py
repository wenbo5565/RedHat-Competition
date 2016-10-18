# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 21:28:17 2016

@author: wenbma
"""

"""
This is the script to Udacity's Nano Degress Program: Machine Learning Engineer. This
problem and datset are from Kaggle competition: RedHat
"""

import pandas as pd
import numpy as np
## import xgboost as xgb

""" Import Data """
people_raw = people = pd.read_csv(r"C:\Users\wenbma\Desktop\Others\Kaggle\RedHat\people.csv\people.csv")
train_raw = pd.read_csv(r"C:\Users\wenbma\Desktop\Others\Kaggle\RedHat\act_train.csv\act_train.csv")

""" Merge people and train by people_id """
train = pd.merge(train_raw,people_raw,how='left',on=['people_id'],suffixes=('_train','_people'))

train.dtypes

for column in train:
    if train[column].dtypes == bool:
          train[column] = train[column].astype(int) 
        
train.dtypes

""" Evaluate number of categories for each categorical variable """
for column in train:
    if train[column].dtypes == object:
        print(train[column].name)        
        print(train[column].unique().size)
        
""" Drop people_id, activity_id, group_1, char_10_train as there are two many categories within them """
train_drop1 = train.drop(['people_id','activity_id','group_1','char_10_train',],axis=1)

""" 
Drop date_train and date_people columns because there are too many categories within them as well. In addition, I cannot be 
sure whether they have predictive power and date variable is always needs to be transformed to other type. So let me remove it 
first and add back later after we quickly build our first model
"""     

train_drop2 = train_drop1.drop(['date_train','date_people'],axis=1)

""" One-Hot Encoder for Categorical Variable"""
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
label = preprocessing.LabelEncoder()



"""Extract outcome from train_drop2"""
all_y = train_drop2[['outcome']]
all_x = train_drop2.drop(['outcome'],axis=1)

for col in all_x:
    if all_x[col].dtypes==object:
        all_x[col]=label.fit_transform(all_x[col])
    
enc.fit(all_x)
all_x_new = enc.transform(all_x)
"""Split dataset into training and validation set by 70% and 30%"""
import random
random.seed(1000)
ind = random.sample(range(0,len(all_y)),int(len(all_y)*0.7))

train_x = all_x_new[ind,]
train_y = all_y.loc[ind,]

mask = np.ones(len(all_y),dtype=bool)
mask[ind]=False

valid_x = all_x_new[mask,]
valid_y = all_y.loc[mask,]
"""Fit Gradient Boosting"""
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(n_estimators=100,learning_rate=0.5,random_state=0).fit(train_x,train_y)

gbm.score(valid_x.toarray(),valid_y)

y_hat=gbm.predict(valid_x.toarray())

y_train_hat = gbm.predict(train_x.toarray())

from sklearn.metrics import roc_auc_score

roc_auc_score(valid_y,y_hat)
roc_auc_score(train_y,y_train_hat)

""" 
Since the auc for train and valid set are very close.
The model is less likely suffering from high variance. So to imporve the model, we need to add
more features back
"""

"""
At this point, we add back char_10_train first as it has the least number of categories among all the
features removed earlier so that it less likely to blow up the memory for one-hot encode
"""

train_drop1 = train.drop(['people_id','activity_id','group_1'],axis=1)

train_drop2 = train_drop1.drop(['date_train','date_people'],axis=1)

""" One-Hot Encoder for Categorical Variable"""
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
label = preprocessing.LabelEncoder()

"""Extract outcome from train_drop2"""
all_y = train_drop2[['outcome']]
all_x = train_drop2.drop(['outcome'],axis=1)

for col in all_x:
    if all_x[col].dtypes==object:
        all_x[col]=label.fit_transform(all_x[col])
    
enc.fit(all_x)
all_x_new = enc.transform(all_x)
"""Split dataset into training and validation set by 70% and 30%"""
import random
random.seed(1000)
ind = random.sample(range(0,len(all_y)),int(len(all_y)*0.7))

train_x = all_x_new[ind,]
train_y = all_y.loc[ind,]

mask = np.ones(len(all_y),dtype=bool)
mask[ind]=False

valid_x = all_x_new[mask,]
valid_y = all_y.loc[mask,]
"""Fit Gradient Boosting"""
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(n_estimators=100,learning_rate=0.5,random_state=0).fit(train_x,train_y)

"""
transfrom sparse matrix to dense matrix trunk by trunk because transformation all at a time blow up
the memory
"""

def chunks(index_array, chunk_size):
    for i in xrange(0,len(index_array),chunk_size):
        yield index_array[i:i+chunk_size]
        
list_ind = list(chunks(range(0,valid_x.shape[0]),200))
list_train_ind = list(chunks(range(0,train_x.shape[0]),200))

y_hat_valid=[]
for i in xrange(0,len(list_ind)):
    y_hat_valid.extend(gbm.predict(valid_x[list_ind[i],:].toarray()).tolist())
    
y_train_hat=[]
for i in xrange(0,len(list_train_ind)):
    y_train_hat.extend(gbm.predict(train_x[list_train_ind[i],:].toarray()).tolist())

from sklearn.metrics import roc_auc_score
roc_auc_score(valid_y,y_hat_valid)
roc_auc_score(train_y,y_train_hat)

"""
-------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------
"""

"""
Add Back Group1 this time
"""

train_drop1 = train.drop(['people_id','activity_id',],axis=1)

train_drop2 = train_drop1.drop(['date_train','date_people'],axis=1)

""" One-Hot Encoder for Categorical Variable"""
from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
label = preprocessing.LabelEncoder()

"""Extract outcome from train_drop2"""
all_y = train_drop2[['outcome']]
all_x = train_drop2.drop(['outcome'],axis=1)

for col in all_x:
    if all_x[col].dtypes==object:
        all_x[col]=label.fit_transform(all_x[col])
    
enc.fit(all_x)
all_x_new = enc.transform(all_x)
"""Split dataset into training and validation set by 70% and 30%"""
import random
random.seed(10000)
ind = random.sample(range(0,len(all_y)),int(len(all_y)*0.7))

train_x = all_x_new[ind,]
train_y = all_y.loc[ind,]

mask = np.ones(len(all_y),dtype=bool)
mask[ind]=False

valid_x = all_x_new[mask,]
valid_y = all_y.loc[mask,]
"""Fit Gradient Boosting"""
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(n_estimators=100,learning_rate=0.5,random_state=0).fit(train_x,train_y)

"""
transfrom sparse matrix to dense matrix trunk by trunk because transformation all at a time blow up
the memory
"""

def chunks(index_array, chunk_size):
    for i in xrange(0,len(index_array),chunk_size):
        yield index_array[i:i+chunk_size]
        
list_ind = list(chunks(range(0,valid_x.shape[0]),200))
list_train_ind = list(chunks(range(0,train_x.shape[0]),200))

y_hat_valid=[]
for i in xrange(0,len(list_ind)):
    y_hat_valid.extend(gbm.predict(valid_x[list_ind[i],:].toarray()).tolist())
    
y_train_hat=[]
for i in xrange(0,len(list_train_ind)):
    y_train_hat.extend(gbm.predict(train_x[list_train_ind[i],:].toarray()).tolist())

from sklearn.metrics import roc_auc_score
roc_auc_score(valid_y,y_hat_valid)
roc_auc_score(train_y,y_train_hat)



