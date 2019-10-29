"""
lambdata-pdugovich - A collection of Data Science helper functions.
"""

import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split


#sample code
ONES = pd.DataFrame(np)
ZEROES = pd.DataFrame(np.zeros(50))


#sample functions
def increment(x):
    return(x + 1)

#fxn to split data into train, validation, and test sets
# def train_validation_test_split(
#     X, y, train_size=0.7, val_size=0.1, test_size=0.2,
#     random_state=None, shuffle=True):

#     assert train_size + val_size + test_size == 1

#     X_train_val, X_test, y_train_val, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_val, y_train_val, test_size=val_size/(train_size+val_size),
#         random_state=random_state, shuffle=shuffle)
    
#     return X_train, X_val, X_test, y_train, y_val, y_test