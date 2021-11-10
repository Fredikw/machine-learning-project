'''
Utility function 

'''
import numpy as np

def to_one_hot_enc(arr):
    
    one_hot_enc = []

    for element in arr:
        
        sample = np.array([0, 0, 0, 0])

        sample[int(element)] = 1

        one_hot_enc.append(sample)
    
    return np.array(one_hot_enc)


def from_one_hot_enc(arr):

    lst = []
    
    for element in arr:

        idx, = np.where(element == 1)

        lst.append(idx[0])

    return np.array(lst)


def reshape_images(arr):

    img_list = []

    for img in arr:
        img = np.reshape(np.array(img),(50,50))
        img_list.append(img)

    return np.array(img_list)


def samples_in_class_count(arr):

    samples_in_class = np.bincount(arr.astype(int))

    return samples_in_class[0], samples_in_class[1], samples_in_class[2], samples_in_class[3]

'''
Preparing data 

'''
import numpy as np
from os import getcwd

x_data = np.load(getcwd() + "/training_set/Xtrain_Classification_Part2.npy") # x_data.shape: (7366, 2500)
y_data = np.load(getcwd() + "/training_set/Ytrain_Classification_Part2.npy") # y_data.shape: (7366,)

'''
Evaluate data

'''

# class1, class2, class3, class4 = samples_in_class_count(y_data)

# total = class1 + class2 + class3 + class4

# # The training set is unbalanced

# print('Share of ..: {:.2f}%'.format(100*class1/total))
# print('Share of ..: {:.2f}%'.format(100*class2/total))
# print('Share of ..: {:.2f}%'.format(100*class3/total))
# print('Share of ..: {:.2f}%'.format(100*class4/total))

'''
confusion matrix

'''

# TODO

'''
Show image

'''

from matplotlib.pyplot import show, imshow

def show_img(img):
    img = np.reshape(np.array(img),(50,50))
    imshow(img,cmap='gray')
    show()

print(y_data[10])

show_img(x_data[10])

# for idx, img in enumerate(y_data):
    
#     if img == 0.0:
#         show_im(x_data[idx])

#     break

        





