import pandas as pd
import numpy as np
from sklearn import preprocessing

def read_data(data_title, col_array):

    # import the data
    data = pd.read_csv(data_title, sep=',', skipinitialspace=True)

    # eliminate the columns that are not useful (categorical columns)
    my_data = data.drop(columns=col_array, axis=1)

    # transform the data into a numpy array
    # x_data = my_data.values

    return my_data

def clean_data(data, col_target):

    # fix values for the entries that are ?
    data.replace(to_replace='?', value=np.nan)

    # fill out the numerical columns with the mean value
    data.fillna(data.mean())

    # transform non-numeric columns into numerical columns
    for column in data.columns:
        if data[column].dtype == np.number:
            continue
        data[column] = preprocessing.LabelEncoder().fit_transform(data[column])

    # split the data removing the column that represents the target (ex: in chronic kidney disease it's classification)
    X = data.drop(col_target, axis=1)

    # target
    Y = data[col_target]

    # Feature scaling
    x_scaler = preprocessing.MinMaxScaler()
    x_scaler.fit(X)
    column_names = X.columns
    X[column_names] = x_scaler.transform(X)

    return X, Y


def visualize_data():
    pass

def pca():
    pass

def svm():
    pass

def neural_networks():
    pass

def bayes_classifier():
    pass

def split():
    pass

def cross_validation():
    pass

def train():
    pass

def validate():
    pass