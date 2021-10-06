import numpy as np
import os

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

slice_at = 90

# Data load
Xdata = np.load(os.getcwd() + "/regression/training_set/Xtrain_Regression_Part1.npy")
Ydata = np.load(os.getcwd() + "/regression/training_set/Ytrain_Regression_Part1.npy")
testXdata = np.load(os.getcwd()+ "/regression/test_set/Xtest_Regression_Part1.npy")

# Linear model
# Training
XdataTraining = Xdata[:slice_at]
YdataTraining = Ydata[:slice_at]
linearModel = linear_model.LinearRegression()
linearModel.fit(XdataTraining,YdataTraining)

# Validation
XdataValidation = Xdata[slice_at:]
YdataValidation = Ydata[slice_at:]

prediction = linearModel.predict(XdataValidation)
error = mean_squared_error(prediction,YdataValidation)     # Linear model yields a small error of Approx. 0.024. This indicates that the model makes good predictions

# Testing
testPrediction = linearModel.predict(testXdata)

#np.save('data.npy', testOutput)

# TODO Try different models
# TODO Find features with linear correlation