from os import getcwd
from utils import *

import numpy as np
import pandas as pd

from matplotlib.pyplot import show
# from sklearn.linear_model import SGDClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (GridSearchCV, cross_val_score, train_test_split)
from sklearn.metrics import balanced_accuracy_score


'''
Preparing data 

'''

x_data = np.load(getcwd() + "/training_set/Xtrain_Classification_Part1.npy") # (6470, 2500)
y_data = np.load(getcwd() + "/training_set/Ytrain_Classification_Part1.npy") # (6470,)

x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(x_data, y_data, test_size=0.10) # x_training_set.shape : (5823, 2500) x_test_set.shape : (647, 2500)


'''
Evaluate data

'''

# num_of_men, num_of_women = num_samples_in_classes(y_data)

# total = num_of_men + num_of_women

# print('Share of men:   {:.2f}%'.format(num_of_men/total))
# print('Share of women: {:.2f}%'.format(num_of_women/total))
# print('Did you just assume my gender?')

'''
Show image

'''
# import matplotlib.pyplot as plt

# def imshow(img):
#     img = np.reshape(np.array(img),(50,50))
#     plt.imshow(img,cmap='gray')
#     show()

# print(y_data[10])
# imshow(x_data[10])

'''
Classification models

'''

# SGD = SGDClassifier(loss='squared_hinge', max_iter = 1000, tol=1e-3,penalty = "l1")

# SGD_score = cross_val_score(SGD, x_data, y_data, scoring='accuracy', cv=5)    # performance: [0.83269378 0.85188028 0.84497314 0.83947773 0.83256528]


# KNN = KNeighborsClassifier(n_neighbors=17)

# KNN_score = cross_val_score(KNN, x_data, y_data, scoring='accuracy', cv=5)    # performance: [0.74111283 0.73415765 0.73647604 0.7187017  0.7187017 ]


# # Multi-layer Perceptron

# y_data = to_one_hot_enc(y_data)

# MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)    # performance: [0.56955178 0.58268934 0.57418856 0.57341577 0.56027821]

# MLP_score = cross_val_score(MLP, x_data, y_data, scoring='accuracy', cv=5)


# # CNN classifier - Se cnn.ipynb for final implementation


'''
Tuning classifier

'''

# # Tuning SGD

# params = {
#     "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
#     "alpha" : [0.0001, 0.001, 0.01, 0.1],
#     "penalty" : ["l2", "l1", "none"],
# }

# SGD_model = GridSearchCV(SGD, params) # {'alpha': 0.0001, 'loss': 'squared_hinge', 'penalty': 'l1'}
# SGD_model.fit(x_data, y_data)

# print(SGD_model.best_params_)


# # Tuning KNN

# k_range = list(range(1, 31))

# params = {
#     "n_neighbors" : k_range
# }

# # defining parameter range
# KNN_model = GridSearchCV(KNN, params, cv=10, scoring='accuracy', return_train_score=False,verbose=1)

# # fitting the model for grid search
# grid_search=KNN_model.fit(x_data, y_data)


'''
confusion matrix

'''
# from seaborn import heatmap
# from sklearn.metrics import confusion_matrix

# SGD.fit(x_training_set, y_training_set)

# predict_training  = SGD.predict(x_training_set)
# predict_test      = SGD.predict(x_test_set)

# TODO Confusion matrix on the wrong format 
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
