import numpy as np

def to_one_hot_enc(arr):

    one_hot_enc = []

    for element in arr:

        if element:
            one_hot_enc.append([1,0])
        else:
            one_hot_enc.append([0,1])
    
    return np.array(one_hot_enc)

# TODO should return array
def from_one_hot_enc(arr):

    not_one_hot_enc = []
    
    for element in arr:

        if element[0]:
            not_one_hot_enc.append(1)
        else:
            not_one_hot_enc.append(0)
    
    return not_one_hot_enc
