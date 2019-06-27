# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:36:23 2019

@author: SEANKurian
"""

import pandas as pd
from sklearn import preprocessing 

df = pd.read_csv(r'C:\Users\SEANKurian\Desktop\attritionSheet.csv')
print(df)
df_encoded = df.apply(preprocessing.LabelEncoder().fit_transform)
print (df_encoded)