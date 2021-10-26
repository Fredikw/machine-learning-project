import numpy as np
import matplotlib.pyplot as plt

from os import getcwd

'''
Loading data

'''

x_data = np.load(getcwd() + "/training_set/Xtrain_Classification_Part1.npy")
y_data = np.load(getcwd() + "/training_set/Ytrain_Classification_Part1.npy")

print(x_data.shape)
print(y_data.shape)


'''
Show image

'''

# def imshow(img):
#     img = np.reshape(np.array(img),(50,50))
#     plt.imshow(img,cmap='gray')
#     plt.show()

# imshow(x_data[2032])

'''
confusion matrix

'''

'''
Utility function

'''

def to_one_hot_enc(arr):

    one_hot_enc = []

    for element in arr:

        if arr:
            one_hot_enc.append(np.array([1,0]))
        else:
            one_hot_enc.append(np.array([0,1]))
    
    return one_hot_enc


def from_one_hot_enc(arr):

    not_one_hot_enc = []
    
    for element in arr:

        if element[0]:
            not_one_hot_enc.append(1)
        else:
            not_one_hot_enc.append(0)
    
    return not_one_hot_enc
