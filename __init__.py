"""
Class version of Data Science helper functions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# sample code
ONES = pd.DataFrame(np.ones(50))
ZEROES = pd.DataFrame(np.zeros(50))
SAMPLE_DATAFRAME = pd.DataFrame(
    {
        'num_legs': [2, 4, 8, 0],
        'num_wings': [2, 0, 0, 0],
        'num_specimen_seen': [10, 2, 1, 8]
        },
    index=['falcon', 'dog', 'spider', 'fish']
)


# sample functions
def increment(x):
    """
    Add 1 to an integer
    """
    return(x + 1)


# fxn to split data into train, validation, and test sets
def train_validation_test_split(
        X, y, train_size=0.7, val_size=0.1, test_size=0.2, random_state=None,
        shuffle=True):
        """
        Given features(X) and target(y), split data into train, val, test sets
        """
        assert train_size + val_size + test_size == 1

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            shuffle=shuffle)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size/(train_size+val_size),
            random_state=random_state, shuffle=shuffle)

        return X_train, X_val, X_test, y_train, y_val, y_test


# function to
def print_nulls(DataFrame):
    """
    Given a dataframe, returns a dataframe with the number of nulls in each
    column listed in descending order
    """
    # Copies dataframe
    df_isnull = DataFrame.copy()
    # Resets index of the sum of nulls for the dataframe
    df_isnull = pd.Series(df_isnull.isnull().sum()).reset_index()
    # Renames columns
    df_isnull = df_isnull.rename(
        columns={'index': 'Column', 0: 'Number of Nulls'})
    # Sorts by highest number of nulls
    df_isnull.sort_values(by='Number of Nulls', ascending=False)
    return print(df_isnull)


class Split:
    """
    Split data into train, validate, and test sets
    """

    def __init__(self, X, y, train_size=0.7, val_size=0.1,
                 test_size=0.2, random_state=None, shuffle=True):
        self.X = X
        self.y = y
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle

    def train_test_validation_split(self):
        """
        Given features and target, splits data into train, test, and validation
        sets
        """
        assert self.train_size + self.val_size + self.test_size == 1
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X,
            self.y,
            self.test_size,
            self.random_state,
            self.shuffle
            )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=self.val_size / (self.train_size/self.val_size)
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
