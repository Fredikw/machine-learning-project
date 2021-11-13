'''
Utility function 

'''
import numpy as np

from matplotlib.pyplot import show, imshow


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


def num_samples_in_classes(arr):

    samples_in_class = np.bincount(arr.astype(int))

    return samples_in_class[0], samples_in_class[1], samples_in_class[2], samples_in_class[3]


def show_img(img):
    img = np.reshape(np.array(img),(50,50))
    imshow(img,cmap='gray')
    show()


def get_indices(arr, val):
    idxs = np.where(arr == val)[0]
    return idxs


def balance_set(x_data, y_data):

  occurrence_lst = num_samples_in_classes(y_data)

  max_samples_of_class = max(occurrence_lst)

  y_data_balanced = []
  x_data_balanced = []

  for clas, occurrence in enumerate(occurrence_lst):
  
    sample_idxs = get_indices(y_data, clas)

    multiplier = int(max_samples_of_class/occurrence)
  
    for sample_idx in sample_idxs:

      temp_lst_x = [x_data[sample_idx]]*multiplier
      temp_lst_y = [y_data[sample_idx]]*multiplier

      x_data_balanced += temp_lst_x
      y_data_balanced += temp_lst_y

  x_data_balanced = np.array(x_data_balanced)
  y_data_balanced = np.array(y_data_balanced)
  
  return x_data_balanced, y_data_balanced


'''
Loading and reading data 

'''

import numpy as np
from os import getcwd

x_data = np.load(getcwd() + "/training_set/Xtrain_Classification_Part2.npy") # x_data.shape: (7366, 2500)
y_data = np.load(getcwd() + "/training_set/Ytrain_Classification_Part2.npy") # y_data.shape: (7366,)


# # Evaluate data

# class1, class2, class3, class4 = num_samples_in_classes(y_data)

# total = class1 + class2 + class3 + class4

# # The training set is unbalanced

# print('Share of Class 1: {:.2f}%'.format(100*class1/total))
# print('Share of Class 2: {:.2f}%'.format(100*class2/total))
# print('Share of Class 3: {:.2f}%'.format(100*class3/total))
# print('Share of Class 4: {:.2f}%'.format(100*class4/total))

# # Share of Class 1: 60.79%
# # Share of Class 2: 4.63%
# # Share of Class 3: 18.16%
# # Share of Class 4: 16.41%


'''
Prepare data

'''
from imblearn.over_sampling import SMOTE

# # Add copies of underrepresented classes - naive approach

# x_data_balanced, y_data_balanced = balance_set(x_data, y_data)


# # Add synthetic samples with SMOTE - not an popular in image processing

sm = SMOTE(random_state=42)

x_smote, y_smote = sm.fit_resample(x_data, y_data)