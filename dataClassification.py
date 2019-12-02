import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns


def load_data(path, flag=True): # Pastorino & Riera i Marin
    """"If flag is True, we are working with data_banknote_authentication.txt file
    If flag is False, we are working with kidney_disease.csv file"""
    if (flag == True):
        columns = ["var", "skewness", "curtosis", "entropy", "class"]
        data = pd.read_csv(path, header=None, names=columns)
    else:
        data = pd.read_csv(path)
    return data

def col_target_def(flag):
    """TODO: decides which columns need to be deleted"""
    """It selects which columns need to be deleted depending on the data we are reading
    If flag is true, we are working with kidney_disease.csv file
    If flag is false, we are working with data_banknote_authentication.txt file"""

    if flag=="true":
        col_target = ["rbc", "pc", "pcc", "ba", "dm", "cad", "appet", "pe", "ane", "classification"]
        return col_target
    return None


def clean_data(data, flag):  # Rodriguez
    if (flag == True):
        x = data.drop(['class'], axis=1)
        y = data['class']
    else:
        ## gustavo's part of kidney disease

    """Cleans the data already read in the load_data() function
    data: data returned from load_data()
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


def visualize_data(data, column):  # Pastorino & Riera i Marin
    """Visualizes in a Boxplot the data once cleaned"""
    print(data.describe())
    sns.boxplot(data = data)
    sns.pairplot(data)
    #column = ["var", "skewness", "curtosis", "entropy"]
    n = len(column)
    #vis = np.zeros(n)
    f, ax = plt.subplots(1, n, figsize=(10,3))
    for i in range(n):
        vis = sns.distplot(data[column[i]],bins=10, ax= ax[i])
        f.savefig('subplot.png')
    return


def pca():  # Moro
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
