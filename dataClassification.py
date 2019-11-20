import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns


def read_data(data_path): # for windows we need to put r before data_path 
    """TODO: review if we delete here the column or better all in the clean_data()"""
    """Reads the data of the file and returns it
    data_title: path and name of the file desired to read
    col_array: columns of the array that can be deleted (in general, the first column)"""

    # import the data
    my_data = pd.read_csv(data_path, sep=',', skipinitialspace=True) 

    
  

    # transform the data into a numpy array
    # x_data = my_data.values

    return my_data


def col_target_def(flag):
    """TODO: decides which columns need to be deleted"""
    """It selects which columns need to be deleted depending on the data we are reading
    If flag is true, we are working with kidney_disease.csv file
    If flag is false, we are working with data_banknote_authentication.txt file"""

    if flag:
        col_target = ["rbc", "pc", "pcc", "ba", "dm", "cad", "appet", "pe", "ane", "classification"]
    else:
        col_target = None

    return col_target


def clean_data(data, col_target):
    """Cleans the data already read in the read_data() function
    data: data returned from read_data()
    col_target: columns non numerical that must be cleaned"""

    # fix values for the entries that are ?
    data.replace(to_replace='?', value=np.nan)

    # fill out the numerical columns with the mean value
    data.fillna(data.mean())

    # delete the rows that have NaN values
    data = data.dropna()

    # transform non-numeric columns into numerical columns
    for column in data.columns:
        if data[column].dtype == np.number:
            continue
        data[column] = preprocessing.LabelEncoder().fit_transform(data[column])

    if col_target is not None:
        # split the data removing the column that represents the target
        # (ex: in chronic kidney disease it's classification)
        X = data.drop(col_target, axis=1)

        # target
        Y = data[col_target]

        # Feature scaling
        x_scaler = preprocessing.MinMaxScaler()
        x_scaler.fit(X)
        column_names = X.columns
        X[column_names] = x_scaler.transform(X)

        return   pd.concat([X,Y], axis=1, sort=False) # to return a cleaned data frame with to same columns as the original 

    elif col_target is None:
        X = data
        # Feature scaling
        x_scaler = preprocessing.MinMaxScaler()
        x_scaler.fit(X)
        column_names = X.columns
        X[column_names] = x_scaler.transform(X)

        return pd.concat([X,None], axis=1, sort=False)


def visualize_data(data):
    """Visualizes in a Boxplot the data once cleaned"""
    data.describe()
    sns.boxplot(data = data)


def pca():
    """TODO: """
    pass


def svm():
    """TODO: """
    pass


def neural_networks():
    """TODO: """
    pass


def bayes_classifier():
    """TODO: """
    pass


def split():
    """TODO: """
    pass


def cross_validation():
    """TODO: """
    pass


def train():
    """TODO: """
    pass


def validate():
    """TODO: """
    pass
