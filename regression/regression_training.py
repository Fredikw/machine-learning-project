import numpy as np
import os
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

slice_at = 90

# Data load
Xdata = np.load(os.getcwd() + "/regression/training_set/Xtrain_Regression_Part1.npy")
Ydata = np.load(os.getcwd() + "/regression/training_set/Ytrain_Regression_Part1.npy")
testXdata = np.load(os.getcwd()+ "/regression/test_set/Xtest_Regression_Part1.npy")

# Training
XdataTraining = Xdata[:slice_at]
YdataTraining = Ydata[:slice_at]
trainingReg = linear_model.LinearRegression()
trainingReg.fit(XdataTraining,YdataTraining)

# Validation [NOT IN USE]
XdataValidation = Xdata[slice_at:]
YdataValidation = Ydata[slice_at:]
validationReg = linear_model.LinearRegression()
validationReg.fit(XdataValidation,YdataValidation)

# Testing
testXdata = testXdata[:slice_at]
testOutput = trainingReg.predict(testXdata)

# Error checking
error = mean_squared_error(YdataTraining,testOutput)
print(error)