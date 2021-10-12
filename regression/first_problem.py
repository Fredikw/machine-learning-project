
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

x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(x_data, y_data, test_size=10)

'''
Correlation matrix for input data

'''
'''
data = np.c_[ x_data, y_data ] 
df = pd.DataFrame(data)
corrMatrix = df.corr()
heatmap(corrMatrix, annot=True)
show()
'''

'''
Regression models

'''

#x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(x_data, y_data, test_size=10)

linearModel = LinearRegression()

XGBoostModel = XGBRegressor()

CatBoostModel = CatBoostRegressor(loss_function='RMSE')

SGDRegressorModel = SGDRegressor()

#Best value alpha = 0.0
KernelRidgeModel = KernelRidge(alpha=1.0)

ElasticNetModel = ElasticNet(random_state=0)

#Better than linear 43 percent of the times
BayesianRidgeModel = BayesianRidge()

GBRegressorModel = GradientBoostingRegressor(learning_rate=0.01, max_depth=1, n_estimators=3000, subsample=0.2)

SVRModel = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

'''
Cross validation

'''
'''
lienar_score                = cross_val_score(linearModel, x_data, y_data, scoring='neg_mean_squared_error', cv=5)
XGBoostModel_score          = cross_val_score(XGBoostModel, x_data, y_data, scoring='neg_mean_squared_error', cv=5)
SGDRegressorModel_score     = cross_val_score(SGDRegressorModel, x_data, np.ravel(y_data), scoring='neg_mean_squared_error', cv=5)
KernelRidgeModel_score      = cross_val_score(KernelRidgeModel, x_data, y_data, scoring='neg_mean_squared_error', cv=5)
ElasticNetModel_score       = cross_val_score(ElasticNetModel, x_data, y_data, scoring='neg_mean_squared_error', cv=5)
BayesianRidgeModel_score    = cross_val_score(BayesianRidgeModel, x_data, np.ravel(y_data), scoring='neg_mean_squared_error', cv=5)
GBRegressorModel_score      = cross_val_score(GBRegressorModel, x_data, np.ravel(y_data), scoring='neg_mean_squared_error', cv=5)
SVRModel_score              = cross_val_score(SVRModel, x_data, np.ravel(y_data), scoring='neg_mean_squared_error', cv=5)
'''

# testXdata = np.load(os.getcwd()+ "/regression/test_set/Xtest_Regression_Part1.npy")
# np.save('data.npy', testOutput)

parameters = {'learning_rate': [0.01,0.02,0.03,0.04, 0.8, 0.1, 0.5],
              'subsample'    : [1.0 , 0.9, 0.5, 0.2, 0.1],
              'n_estimators' : [100,500,1000, 1500, 2000, 3000],
              'max_depth'    : [1,2,3,4,6,8,10]
             }


linearModel.fit(x_training_set, y_training_set)
y_pred_lin = linearModel.predict(x_test_set)
print(mean_squared_error(y_test_set, y_pred_lin))

GBRegressorModel.fit(x_training_set, np.ravel(y_training_set))
y_pred = GBRegressorModel.predict(x_test_set)
print(mean_squared_error(y_test_set, y_pred))


# GBRegressorModel.get_params()
'''
             {'alpha': 0.9, 
             'ccp_alpha': 0.0,
             'criterion': 'friedman_mse',
             'init': None,
             'learning_rate': 0.1,
             'loss': 'squared_error',
             'max_depth': 3,
             'max_features': None,
             'max_leaf_nodes': None, 
             'min_impurity_decrease': 0.0, 
             'min_samples_leaf': 1, 
             'min_samples_split': 2, 
             'min_weight_fraction_leaf': 0.0,
             'n_estimators': 100,
             'n_iter_no_change': None,
             'random_state': 0,
             'subsample': 1.0,
             'tol': 0.0001,
             'validation_fraction': 0.1,
             'verbose': 0,
             'warm_start': False}
             '''