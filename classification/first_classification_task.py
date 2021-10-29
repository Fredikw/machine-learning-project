from os import getcwd

import numpy as np
import pandas as pd

from matplotlib.pyplot import show
from seaborn import heatmap
from utils import *
# import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV


'''
Loading data

'''

x_data = np.load(getcwd() + "/training_set/Xtrain_Classification_Part1.npy") # (6513, 2500)
y_data = np.load(getcwd() + "/training_set/Ytrain_Classification_Part1.npy") # (6513,)

x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(x_data, y_data, test_size=0.10)

'''
Show image

'''

# def imshow(img):
#     img = np.reshape(np.array(img),(50,50))
#     plt.imshow(img,cmap='gray')
#     plt.show()

# imshow(x_data[2032])

'''
Classification models

'''

# SGD = SGDClassifier(loss='squared_hinge', max_iter = 1000, tol=1e-3,penalty = "l1")

# # With cross validation

# SGD_score = cross_val_score(SGD, x_data, y_data, scoring='accuracy', cv=5) # performance: [0.83269378 0.85188028 0.84497314 0.83947773 0.83256528]

KNN = KNeighborsClassifier()

# KNN_score = cross_val_score(KNN, x_data, y_data, scoring='accuracy', cv=5)

# print(KNN_score)

'''
Tuning classifier


'''

# # Tuning SGD


# params = {
#     "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
#     "alpha" : [0.0001, 0.001, 0.01, 0.1],
#     "penalty" : ["l2", "l1", "none"],
# }

# model = GridSearchCV(SGD, params) # {'alpha': 0.0001, 'loss': 'squared_hinge', 'penalty': 'l1'}
# model.fit(x_data, y_data)

# print(model.best_params_)

# # Tuning KNN

k_range = list(range(1, 31))

params = {
    "n_neighbors" : k_range,
    "weights" : ['uniform', 'distance'],
    "algorithm" : ['auto', 'ball_tree', 'kd_tree', 'brute'],
    "leaf_size" : [20, 30, 40],
    "p" : [1, 2]
}

# defining parameter range
grid = GridSearchCV(KNN, params, cv=10, scoring='accuracy', return_train_score=False,verbose=1)

# fitting the model for grid search
grid_search=grid.fit(x_data, y_data)

print(grid_search.best_params_)


'''
confusion matrix

'''

# SGD.fit(x_training_set, y_training_set)

# predict_training  = SGD.predict(x_training_set)
# predict_test      = SGD.predict(x_test_set)

# # print(confusion_matrix(y_training_set, predict_training, labels=[0.0, 1.0]))
# # print(confusion_matrix(y_test_set, predict_test, labels=[0.0, 1.0]))

# data_training = {'Target training': y_training_set, 'Predicted training': predict_training}
# data_test     = {'Target test': y_test_set, 'Predicted test': predict_test}

# df_training   = pd.DataFrame(data_training, columns=['Target training','Predicted training'])
# df_test       = pd.DataFrame(data_test, columns=['Target test','Predicted test'])

# confusion_matrix_training  = pd.crosstab(df_training['Target training'], df_training['Predicted training'],
#                                          rownames=['Target class'], colnames=['Output class'], margins = True, normalize=True)
# confusion_matrix_test      = pd.crosstab(df_test['Target test'], df_test['Predicted test'], rownames=['Target class'],
#                                          colnames=['Output class'], margins = True, normalize=True)

# heatmap(confusion_matrix_training, annot=True)
# show()
# heatmap(confusion_matrix_test, annot=True)
# show()
