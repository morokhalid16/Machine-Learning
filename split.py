# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 03:02:57 2019

@author: MORO
"""

import numpy as np
from sklearn.model_selection import train_test_split
def split(flag,cleaned_data,test_proportion=0.25):
    """ If flag is true, we are working with kidney_disease.csv file
    If flag is false, we are working with data_banknote_authentication.txt file"""
    if flag=="true": 
        n=len(cleaned_data)
        X=cleaned_data
        Y=cleaned_data["classification"]
        del X['classification']
        X_train, X_test,Y_train,Y_test= train_test_split(X,Y,test_size=test_proportion) 
        return X_train,X_test,Y_train,Y_test
       
    else:
        X_train, X_test= train_test_split(cleaned_data, test_size=test_proportion)
        return X_train,X_test