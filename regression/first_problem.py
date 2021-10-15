
import numpy as np

from xgboost import XGBRegressor
from sklearn.svm import SVR
import pandas as pd
from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split

from os import getcwd
from matplotlib.pyplot import show
from seaborn import heatmap

# Data load
x_data = np.load(getcwd() + "/regression/training_set/Xtrain_Regression_Part1.npy")
y_data = np.load(getcwd() + "/regression/training_set/Ytrain_Regression_Part1.npy")

'''
Correlation matrix for input data

'''

# data = np.c_[ x_data, y_data ] 
# df = pd.DataFrame(data)
# corrMatrix = df.corr()
# heatmap(corrMatrix, annot=True)
# show()


'''
Regression models

'''

linearModel = LinearRegression()

# XGBoostModel = XGBRegressor()

# CatBoostModel = CatBoostRegressor(loss_function='RMSE')

# SGDRegressorModel = SGDRegressor()

# #Best value alpha = 0.0
# KernelRidgeModel = KernelRidge(alpha=1.0)

ElasticNetModel = ElasticNet(alpha=0.0001, l1_ratio=0.98)

# BayesianRidgeModel = BayesianRidge()

# GBRegressorModel = GradientBoostingRegressor(learning_rate=0.01, max_depth=1, n_estimators=3000, subsample=0.2)

# SVRModel = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

'''
Tuning the hyper-parameters of an estimator

'''

# from numpy import arange
# from sklearn.model_selection import RepeatedKFold
# from sklearn.linear_model import ElasticNetCV


# # Condig cross validation
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# # grid = dict()
# # grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
# # grid['l1_ratio'] = arange(0, 1, 0.01)

# # # define search
# # elasticNet_search = GridSearchCV(ElasticNetModel, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# # # perform the search
# # results = elasticNet_search.fit(x_data, y_data)
# # # summarize
# # print('Config: %s' % results.best_params_)

# ratios = arange(0, 1, 0.01)
# alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
# model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
# # fit model
# model.fit(x_data, y_data)
# # summarize chosen configuration
# print('alpha: %f' % model.alpha_)
# print('l1_ratio_: %f' % model.l1_ratio_)

'''
Submission

'''

# testXdata = np.load(os.getcwd()+ "/regression/test_set/Xtest_Regression_Part1.npy")
# TODO Save prediction for final regressor
# np.save('data.npy', testOutput)


'''
Testing

'''

# linear_wins = 0
# elastic_wins = 0

# linear_score = cross_val_score(linearModel, x_data, y_data, scoring='neg_mean_squared_error', cv=10)
# elastic_score = cross_val_score(ElasticNetModel, x_data, y_data, scoring='neg_mean_squared_error', cv=10)

# for i in range(len(linear_score)):
#     if linear_score[i] > elastic_score[i]:
#         linear_wins += 1
#     else:
#         elastic_wins += 1

# print('---------------------------------------------------')
# print('Linear wins      ', linear_wins)
# print('elastic_wins     ', elastic_wins)

