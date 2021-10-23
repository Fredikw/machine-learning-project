import numpy as np

def get_unique_elements(arr):
    c = set(arr)
    return np.array(list(c))