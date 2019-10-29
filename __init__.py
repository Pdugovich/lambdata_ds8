"""
lambdata-pdugovich - A collection of Data Science helper functions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#sample code
ONES = pd.DataFrame(np)
ZEROES = pd.DataFrame(np.zeros(50))


#sample functions
def increment(x):
    """Add 1 to an integer"""
    return(x + 1)

# fxn to split data into train, validation, and test sets
def train_validation_test_split(
    X, y, train_size=0.7, val_size=0.1, test_size=0.2,
    random_state=None, shuffle=True):
    """Given features(X) and target(y), split data into train, val, test sets"""

    assert train_size + val_size + test_size == 1

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(train_size+val_size),
        random_state=random_state, shuffle=shuffle)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# function to 
def print_nulls(DataFrame):
    """ Given a dataframe, returns a dataframe with the number of nulls in each
    column listed in descending order"""
    df_isnull = DataFrame.copy()
    df_isnull = pd.Series(df_isnull.isnull().sum()).reset_index()
    df_isnull = df_isnull.rename(columns={'index': 'Column', 0: 'Number of Nulls'})
    df_isnull.sort_values(by='Number of Nulls', ascending=False)
    return print(df_isnull)
