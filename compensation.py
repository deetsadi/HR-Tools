# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:46:17 2019

@author: SEANKurian
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
import os
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv(os.path.join('ec', 'employee-compensation.csv'))
df = df.drop(['Organization Group Code', 'Organization Group', 'Department Code', 'Union Code', 'Union', 'Job Family Code', 'Job Family', 'Job Code', 'Employee Identifier',  ], 1)
X = df[['Department', 'Job']]
y = df['Total Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
