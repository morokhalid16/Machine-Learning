import pandas as pd
from sklearn import preprocessing

def read_data(data_title, col_array):

    # import the data
    data = pd.read_csv(data_title)

    # eliminate the columns that are not useful
    my_data = data.drop(col_array, axis=1)

    # transform the data into a numpy array
    x_data = my_data.values

    return x_data

def clean_data(data):

    # add values for the entries that are zero

    # scale

    std_scale = preprocessing.StandardScaler().fit(data)
    x_scaled = std_scale.transform(data)

    #center


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