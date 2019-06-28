# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:36:23 2019

@author: SEANKurian
"""

import pandas as pd
from sklearn import preprocessing 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\SEANKurian\Desktop\attritionSheet.csv')

#Turning all values into scaled numbers
df_encoded = df.apply(preprocessing.LabelEncoder().fit_transform)

predictors = df.drop(columns=['Attrition'])
target = df['Attrition'].values

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size=0.2, random_state=1, stratify=target)

model = KNeighborsClassifier(n_neighbors=3)
#model.fit()