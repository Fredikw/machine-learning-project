
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import ElasticNet

from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import LocalOutlierFactor

from math import inf

from os import getcwd

# Data load
x_data = np.load(getcwd() + "/regression/training_set/Xtrain_Regression_Part2.npy")
y_data = np.load(getcwd() + "/regression/training_set/Ytrain_Regression_Part2.npy")

'''
Regression models

'''

linearModel = LinearRegression()

# ElasticNetModel = ElasticNet(alpha=0.001, l1_ratio=1)

'''
Removing outliers from data sample

'''

# MSE of raw data
linearModel_score = cross_val_score(linearModel, x_data, y_data, scoring='neg_mean_squared_error', cv=5)
print(linearModel_score)

data = np.c_[ x_data, y_data ]


# # Method 1

# linearModel.fit(x_data, y_data)

# prediciton          = linearModel.predict(x_data)
# distance_from_mean  = prediciton - y_data

# # plt.plot(distance_from_mean, 'ro')
# # plt.show()

# tol = 2

# outlier_list = []

# for idx, distance in enumerate(distance_from_mean):

#     if abs(distance) > tol:
#         outlier_list.append(idx)

# for idx, outlier in enumerate(outlier_list):

#     outlier -= idx

#     x_data = np.delete(x_data, outlier, 0)
#     y_data = np.delete(y_data, outlier, 0)

# linearModel_score = cross_val_score(linearModel, x_data, y_data, scoring='neg_mean_squared_error', cv=5)
# print(linearModel_score)


# # Methode 2

# # LocalOutlierFactor


# Method 3

# # Manual evaluation of outliers

# data = np.c_[ x_data, y_data ]

# for column in range(len(data.T)):

#     y_data = data[:, column]

#     x_data = np.delete(data, column, 1)
    
#     linearModel = LinearRegression()
#     linearModel.fit(x_data, y_data)

#     prediciton          = linearModel.predict(x_data)
#     distance_from_mean  = prediciton - y_data

#     plt.plot(distance_from_mean, 'ro')
#     plt.title('feature ' + str(column))
#     plt.show()

# # Removing outliers

# tol = [3, inf, inf, inf, 3, 2.5, inf, inf, 3, inf, inf, 3, inf, inf, inf, inf, inf, inf, inf, inf, 4]
# # tol = [3, 2, 2, 4, 1.5, 1.5, 2, 2, 2, inf, inf, 1, inf, 2, inf, 2, 2, 2, 2, inf, 2]

# outlier_list = []

# for column in range(len(data.T)):

#     y_data = data[:, column]

#     x_data = np.delete(data, column, 1)
    
#     linearModel = LinearRegression()
#     linearModel.fit(x_data, y_data)

#     prediciton          = linearModel.predict(x_data)
#     distance_from_mean  = prediciton - y_data
    
#     for idx, distance in enumerate(distance_from_mean):

#         if abs(distance) > tol[column]:
#             outlier_list.append(idx)


# for idx, outlier in enumerate(outlier_list):

#     outlier -= idx

#     x_data = np.delete(x_data, outlier, 0)
#     y_data = np.delete(y_data, outlier, 0)