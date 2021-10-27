import numpy as np
# import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


from utils import *
from os import getcwd

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


SGD = SGDClassifier(max_iter = 1000, tol=1e-3,penalty = "elasticnet")

# SGD.fit(x_training_set, y_training_set)

# predict = SGD.predict(x_test_set)

# # With cross validation

# SGD_score = cross_val_score(SGD, x_data, y_data, scoring='accuracy', cv=5)

# print(SGD_score)


'''
confusion matrix

'''

