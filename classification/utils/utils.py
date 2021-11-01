import numpy as np
import matplotlib.pyplot as plt


def to_one_hot_enc(arr):

    one_hot_enc = []

    for element in arr:

        if element:
            one_hot_enc.append([1,0])
        else:
            one_hot_enc.append([0,1])
    
    return np.array(one_hot_enc)


def from_one_hot_enc(arr):

    arr_list = []
    
    for element in arr:

        if element[0]:
            arr_list.append(1)
        else:
            arr_list.append(0)
    
    return np.array(arr_list)


def reshape_images(arr):

    img_list = []

    for img in arr:
        img = np.reshape(np.array(img),(50,50))
        img_list.append(img)

    return np.array(img_list)
