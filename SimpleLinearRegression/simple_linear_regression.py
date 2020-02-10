#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:39:44 2020

@author: priyanka3588
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset 
dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values

from sklearn.model_selection import train_test_split
x_train , x_test, y_train , y_test = train_test_split(x , y , test_size = 1/3 , random_state= 0)

#Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)

#Predicting the test set results
y_pred = regressor.predict(x_test)

#Visualization of the training set results
plt.scatter(x_train , y_train , color='red')
plt.plot(x_train , regressor.predict(x_train) , color='blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualization of the test set results
plt.scatter(x_test , y_test , color='red')
plt.plot(x_train , regressor.predict(x_train) , color='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
 