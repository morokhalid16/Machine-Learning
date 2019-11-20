import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns


def load_data(flag,path): # use ( for windows ) syntax r"path" 
    """"If flag is true, we are working with kidney_disease.csv file
    If flag is false, we are working with data_banknote_authentication.txt file"""
    if flag=="true" :
        return pd.read_csv(path)
    else:
        return pd.read_csv(path,header=None)# dont add header so we dont lose the first row


def col_target_def(flag):
    """TODO: decides which columns need to be deleted"""
    """It selects which columns need to be deleted depending on the data we are reading
    If flag is true, we are working with kidney_disease.csv file
    If flag is false, we are working with data_banknote_authentication.txt file"""

    if flag=="true":
        col_target = ["rbc", "pc", "pcc", "ba", "dm", "cad", "appet", "pe", "ane", "classification"]
        return col_target
    return None

   


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
