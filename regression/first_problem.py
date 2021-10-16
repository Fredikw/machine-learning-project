
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split

from os import getcwd
from seaborn import heatmap

# Data load
x_data = np.load(getcwd() + "/regression/training_set/Xtrain_Regression_Part1.npy")
y_data = np.load(getcwd() + "/regression/training_set/Ytrain_Regression_Part1.npy")

'''
Correlation matrix for input data

'''

data = np.c_[ x_data, y_data ] 
df = pd.DataFrame(data)
corrMatrix = df.corr()
heatmap(corrMatrix, annot=False)
plt.show()

'''
Regression models

'''

linearModel = LinearRegression()

# XGBoostModel = XGBRegressor()

# CatBoostModel = CatBoostRegressor(loss_function='RMSE')

# SGDRegressorModel = SGDRegressor()

# #Best value alpha = 0.0
# KernelRidgeModel = KernelRidge(alpha=1.0)

ElasticNetModel = ElasticNet(alpha=0.001, l1_ratio=1)

# BayesianRidgeModel = BayesianRidge()

# GBRegressorModel = GradientBoostingRegressor(learning_rate=0.01, max_depth=1, n_estimators=3000, subsample=0.2)

# SVRModel = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

'''
Tuning the hyper-parameters of an ElasticNetModel

Found best parameters to be alpha=0.001, l1_ratio=1
'''

# from numpy import arange
# from sklearn.model_selection import RepeatedKFold
# from sklearn.linear_model import ElasticNetCV


# # Condig cross validation
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# ratios = arange(0, 1, 0.01)
# alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
# model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
# # fit model
# model.fit(x_data, y_data)
# # summarize chosen configuration
# print('alpha: %f' % model.alpha_)
# print('l1_ratio_: %f' % model.l1_ratio_)

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

'''
Submission

'''

ElasticNetModel.fit(x_data, y_data)
x_data_test = np.load(getcwd()+ "/regression/test_set/Xtest_Regression_Part1.npy")
test_output = ElasticNetModel.predict(x_data_test)
np.save('regression/test_set_predictions.npy', test_output)

'''
Plotting beta values

'''

plt.plot(ElasticNetModel.coef_, 'ro')
plt.grid()
plt.xlabel('feature i')
plt.ylabel('beta')
plt.title('ElasticNet beta')
plt.show()