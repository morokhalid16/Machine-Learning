# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 03:02:57 2019

@author: MORO
"""

import numpy as np
from sklearn.model_selection import train_test_split
def split(cleand_data,test_proportion=0.25):
    X_train, X_test= train_test_split(cleaned_data, test_size=test_proportion)
    return X_train,X_test