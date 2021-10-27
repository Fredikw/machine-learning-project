import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from utils import *
from os import getcwd
from seaborn import heatmap
from matplotlib.pyplot import show

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

# # With cross validation

# SGD_score = cross_val_score(SGD, x_data, y_data, scoring='accuracy', cv=5)

# print(SGD_score)


'''
confusion matrix

'''


SGD.fit(x_training_set, y_training_set)

predict_training  = SGD.predict(x_training_set)
predict_test      = SGD.predict(x_test_set)

# print(confusion_matrix(y_training_set, predict_training, labels=[0.0, 1.0]))
# print(confusion_matrix(y_test_set, predict_test, labels=[0.0, 1.0]))

data_training = {'Target training': y_training_set, 'Predicted training': predict_training}
data_test     = {'Target test': y_test_set, 'Predicted test': predict_test}

df_training   = pd.DataFrame(data_training, columns=['Target training','Predicted training'])
df_test       = pd.DataFrame(data_test, columns=['Target test','Predicted test'])

confusion_matrix_training  = pd.crosstab(df_training['Target training'], df_training['Predicted training'], rownames=['Target class'], colnames=['Output class'], margins = True, normalize=True)
confusion_matrix_test      = pd.crosstab(df_test['Target test'], df_test['Predicted test'], rownames=['Target class'], colnames=['Output class'], margins = True, normalize=True)


heatmap(confusion_matrix_training, annot=True)
show()
heatmap(confusion_matrix_test, annot=True)
show()