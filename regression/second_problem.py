
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import cross_val_score

from os import getcwd

# Data load
x_data = np.load(getcwd() + "/regression/training_set/Xtrain_Regression_Part2.npy")
y_data = np.load(getcwd() + "/regression/training_set/Ytrain_Regression_Part2.npy")

'''
Regression models

'''

linearModel = LinearRegression()

#ElasticNetModel = ElasticNet(alpha=0.001, l1_ratio=1)

'''
Removing outliers from data sample

'''

# MSE of raw data
linearModel_score = cross_val_score(linearModel, x_data, y_data, scoring='neg_mean_squared_error', cv=5)
print(linearModel_score)

linearModel.fit(x_data, y_data)

prediciton          = linearModel.predict(x_data)
distance_from_mean  = prediciton - y_data

plt.plot(distance_from_mean, 'ro')
plt.show()

tol = 2

outlier_list = []

for idx, distance in enumerate(distance_from_mean):

    if abs(distance) > tol:
        outlier_list.append(idx)

for idx, outlier in enumerate(outlier_list):

    outlier -= idx

    x_data = np.delete(x_data, outlier, 0)
    y_data = np.delete(y_data, outlier, 0)

plt.plot(distance_from_mean, 'ro')
plt.show()

linearModel_score = cross_val_score(linearModel, x_data, y_data, scoring='neg_mean_squared_error', cv=5)
print(linearModel_score)

print(outlier_list)