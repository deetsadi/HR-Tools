# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:36:23 2019

@author: SEANKurian
"""

#Was using train test split on last commit, found cross val score is more accurate
#Plan on using hyper tuning to determine optimal number of neighbors, to be implemented

import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv(r'C:\Users\SEANKurian\Desktop\attritionSheet.csv')

#Turning all values into scaled numbers
df_encoded = df.apply(preprocessing.LabelEncoder().fit_transform)

predictors = df_encoded.drop(columns=['Attrition'])
target = df_encoded['Attrition'].values


model = KNeighborsClassifier(n_neighbors=3)
cv_scores = cross_val_score(model, predictors, target, cv=5)


print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))