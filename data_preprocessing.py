# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 00:37:38 2020

@author: Priyanka
"""

import numpy as np
import matplotlib.pyplot as ply
import pandas as pd

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy= 'mean')
x[: , 1:3] = imputer.fit_transform(x[: , 1:3])

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_x = LabelEncoder()
x[: , 0] = labelEncoder_x.fit_transform(x[: , 0])

#oneHotEncoder_x = OneHotEncoder(categorical_features=[0])
#x = oneHotEncoder_x.fit_transform(x)

#transform = ColumnTransformer([('one_hot_encoder' , OneHotEncoder() , [0])] , remainder='passthrough')
#x = np.array(transform.fit_transform[x])
#print(x)


transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],remainder='passthrough')
x = np.array(transformer.fit_transform(x), dtype=np.float)


labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test =sc_x.transform(x_test)
