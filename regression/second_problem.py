import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import ElasticNet

from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import LocalOutlierFactor

from math import inf
from os import getcwd

# Data load
x_data = np.load(getcwd() + "/training_set/Xtrain_Regression_Part2.npy")
y_data = np.load(getcwd() + "/training_set/Ytrain_Regression_Part2.npy")

'''
Regression models

'''

linearModel = LinearRegression()

# ElasticNetModel = ElasticNet(alpha=0.001, l1_ratio=1)

'''
Removing outliers from data sample

'''


# # MSE of raw data
# linearModel_score = cross_val_score(linearModel, x_data, y_data, scoring='neg_mean_squared_error', cv=5)
# print(linearModel_score)

data = np.c_[x_data, y_data]


# Method 1

linearModel.fit(x_data, y_data)

prediciton          = linearModel.predict(x_data)
distance_from_mean  = prediciton - y_data

# plt.plot(distance_from_mean, 'ro')
# plt.show()

data_std = np.std(distance_from_mean)

standard_diviations = 2
cut_off = data_std * standard_diviations

outliers = []

for idx, distance in enumerate(distance_from_mean):

    if abs(distance) > cut_off:
        outliers.append(idx)

outliers = set(outliers)
outliers = sorted(list(outliers))

print(outliers)

for idx, outlier in enumerate(outliers):

    outlier -= idx

    x_data = np.delete(x_data, outlier, 0)
    y_data = np.delete(y_data, outlier, 0)

linearModel_score = cross_val_score(linearModel, x_data, y_data, scoring='neg_mean_squared_error', cv=5)
print(linearModel_score)


# # Methode 2

# # LocalOutlierFactor


# Method 3

columns = data.shape[1]

outliers = []

# Identify outliers

for idx in range(columns):

    y_data = data[:, idx]
    x_data = np.delete(data, idx, 1)

    linearModel = LinearRegression()
    linearModel.fit(x_data, y_data)

    prediciton          = linearModel.predict(x_data)
    distance_from_mean  = prediciton - y_data

    # Using Standard Deviation Method

    data_mean, data_std = np.mean(distance_from_mean), np.std(distance_from_mean)

    standard_diviations = 3
    cut_off = data_std * standard_diviations

    for idx, distance in enumerate(distance_from_mean):

        if abs(distance) > cut_off:
            outliers.append(idx)

    # plt.plot(distance_from_mean, 'ro')
    # plt.title('feature ' + str(column))
    # plt.show()

outliers = set(outliers)
outliers = sorted(list(outliers))

# Remove outliers

x_data = np.load(getcwd() + "/training_set/Xtrain_Regression_Part2.npy")
y_data = np.load(getcwd() + "/training_set/Ytrain_Regression_Part2.npy")

for idx, outlier in enumerate(outliers):

    outlier -= idx

    x_data = np.delete(x_data, outlier, 0)
    y_data = np.delete(y_data, outlier, 0)

linearModel_score = cross_val_score(linearModel, x_data, y_data, scoring='neg_mean_squared_error', cv=5)
print(linearModel_score)