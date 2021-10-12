
import numpy as np

from xgboost import XGBRegressor
from sklearn.svm import SVR
#from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from os import getcwd

# Data load
x_data = np.load(getcwd() + "/training_set/Xtrain_Regression_Part1.npy")
y_data = np.load(getcwd() + "/training_set/Ytrain_Regression_Part1.npy")

x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(x_data, y_data, test_size=10)

# Linear model

linearModel = LinearRegression()
linearModel.fit(x_training_set,y_training_set)

y_pred_linear = linearModel.predict(x_test_set)
mse_linear = mean_squared_error(y_test_set, y_pred_linear)    #0.009726150151957687

# XGBoost Regressor

XGBoostModel = XGBRegressor(verbosity=0)
XGBoostModel.fit(x_training_set, y_training_set)

y_pred_XGBoost = XGBoostModel.predict(x_test_set)
mse_xgbr = mean_squared_error(y_test_set, y_pred_XGBoost)

# CatBoost Regressor
'''
CatBoostModel = CatBoostRegressor(loss_function='RMSE')
CatBoostModel.fit(x_training_set, y_training_set)

y_pred_CatBoost = CatBoostModel.predict(x_test_set)
mse_catBoost = mean_squared_error(y_test_set, y_pred_CatBoost)
'''
# Stochastic Gradient Descent Regression

SGDRegressorModel = SGDRegressor()
SGDRegressorModel.fit(x_training_set, np.ravel(y_training_set))

y_pred_SGDRegressor = SGDRegressorModel.predict(x_test_set)
mse_SGDRegressor = mean_squared_error(y_test_set, y_pred_SGDRegressor)

# Kernel Ridge Regression

KernelRidgeModel = KernelRidge(alpha=1.0)                   # best value is alpha = 0.0
KernelRidgeModel.fit(x_training_set, y_training_set)

y_pred_KernelRidge = KernelRidgeModel.predict(x_test_set)
mse_KernelRidge = mean_squared_error(y_test_set, y_pred_KernelRidge)


# Elastic Net Regression

ElasticNetModel = ElasticNet(random_state=0)
ElasticNetModel.fit(x_training_set, y_training_set)

y_pred_ElasticNet = ElasticNetModel.predict(x_test_set)
mse_ElasticNet = mean_squared_error(y_test_set, y_pred_ElasticNet)

# Bayesian Ridge Regression                     better than linear 43 percent of the times

BayesianRidgeModel = BayesianRidge()
BayesianRidgeModel.fit(x_training_set, np.ravel( y_training_set))

y_pred_BayesianRidge = BayesianRidgeModel.predict(x_test_set)
mse_BayesianRidge = mean_squared_error(y_test_set, y_pred_BayesianRidge)


# Gradient Boosting Regression

GBRegressorModel = GradientBoostingRegressor(random_state=0)
GBRegressorModel.fit(x_training_set, np.ravel( y_training_set))

y_pred_GradientBoosting = GBRegressorModel.predict(x_test_set)
mse_GradientBoosting = mean_squared_error(y_test_set, y_pred_GradientBoosting)

# Support Vector Machine

SVRModel = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
SVRModel.fit(x_training_set, np.ravel( y_training_set))

y_pred_SVR = SVRModel.predict(x_test_set)
mse_SVR = mean_squared_error(y_test_set, y_pred_SVR)


print('------------------------------MSE of Regressors------------------------------')
print('MSE of linear Regressor:         ', mse_linear)
print('MSE of XGBoost Regressor:        ', mse_xgbr)
#print('MSE of CatBoost Regressor:       ', mse_catBoost)
print('MSE of SGD Regressor:            ', mse_SGDRegressor)
print('MSE of KernelRidge Regressor:    ', mse_KernelRidge)
print('MSE of ElasticNet Regressor:     ', mse_ElasticNet)
print('MSE of BayesianRidge Regressor:  ', mse_BayesianRidge)
print('MSE of GB Regressor:             ', mse_GradientBoosting)
print('MSE of SVR Regressor:            ', mse_SVR)

# TODO Find methodes for evaluation data

# testXdata = np.load(os.getcwd()+ "/regression/test_set/Xtest_Regression_Part1.npy")
# Predict y with best model
# np.save('data.npy', testOutput)
# Use function to evaluate the format of your submitted data