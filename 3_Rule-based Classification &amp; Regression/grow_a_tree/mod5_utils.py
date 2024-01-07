import pandas as pd
import matplotlib as plt
import seaborn as sns
import pandas as pd
import numpy as np
import unittest 

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from unittest.mock import patch


def data_split_test(x_train, y_train, x_test, y_test):
    """
    Task 2: Check correct train-test-split
    
    """
    # Check for train-test split sizes
    if x_train.shape != (426, 30) or x_test.shape != (143,30) or y_train.shape != (426,) or y_test.shape != (143, ):
        print("="*30)
        print(f"Test failed (0/1): Train or test set sizes are not appropriate")
    else:
        print("="*30)
        print(f"Test passed (1/1): Train and test set sizes are appropriate")

def test_model(dt_model, criterion):
    """
    Task 3 & 4: Check correct model implementation
    
    """
    if criterion == "entropy":
        assert dt_model.criterion == "entropy", "DecisionTree model has incorrect splitting criterion."
    else:
        assert dt_model.criterion == "gini", "DecisionTree model has incorrect splitting criterion."
    assert dt_model.max_depth == 5, "DecisionTree model has incorrect max depth."
    
    print("="*30)
    print("Test passed (1/1): All models have been implemented correctly.")