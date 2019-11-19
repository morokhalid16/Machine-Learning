# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:30:05 2019

@author: MORO
"""

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt




def  pca_explained_variance(cleaned_data): #plot a graph to choose the number of dimentions
    pca=PCA().fit(cleaned_data)
    #Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Pulsar Dataset Explained Variance')
    plt.show()
 
    
    
    
#Once we choose the number of dimentsions, we can perform a pca 
def pca(cleaned_data,n_components):
    pca = PCA(n_components)
    dataset = pca.fit_transform(cleaned_data)
    return dataset # returns a new dataset that is ready to be split in train and test data

