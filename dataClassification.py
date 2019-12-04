import pandas as pd
import numpy as np
from sklearn import preprocessing, decomposition
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier¶


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


def clean_data(data, flag):  # Rodrigues
    if (flag == True):
        x = data.drop(['class'], axis=1)
        y = data['class']
    #else: #gustavo's part of kidney disease

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


def pca(cleaned_data, n_components):  # Moro

    std_scale = preprocessing.StandardScaler().fit(cleaned_data)
    x_scaled = std_scale.transform(cleaned_data)
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(x_scaled)
    dataset = pca.transform(x_scaled)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())

    return dataset


def svm(X_train, train_y, X_test, test_y):  # Pastorino & Riera i Marin
    svclassifier = SVC(kernel='linear')
    clf = svclassifier.fit(X_train, train_y)
    y_pred = svclassifier.predict(X_test)
    print(confusion_matrix(test_y, y_pred))
    print(classification_report(test_y, y_pred))
    print(accuracy_score(test_y, y_pred))

    return


"""tested only on banknote authentication for the moment
works well with following values n_layers = 3, n_neurons_per_layer = [4,8,1], kernel_init = 'uniform', activ = 'relu'"""
"""for the chronic kidney disease -> 2 layers, 256 neurons first, 1 neuron last"""

def MLP(n_layers, n_neurons_per_layer, X_train, X_test, test_y, train_y, kernel_init, activ, batch_size, epochs):
    model = Sequential()
    model.add(Dense(n_neurons_per_layer[0], kernel_initializer=kernel_init, activation=activ, input_shape=(X_train.shape[1],)))
    for i in range(1, n_layers-1):
        model.add(Dense(n_neurons_per_layer[i], kernel_initializer=kernel_init))
    model.add(Dense(1, kernel_initializer=kernel_init, activation='sigmoid'))
    model.summary()
    sgd = keras.optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history = model.fit(X_train, train_y, batch_size, epochs)
    np.set_printoptions(precision=4, suppress=True)
    eval_results = model.evaluate(X_test, test_y, verbose=0)
    print("\nLoss, accuracy on test data: ")
    print("%0.4f %0.2f%%" % (eval_results[0], eval_results[1]*100))

    return

"""supposing for now that we have already splitted data in x and target y"""
def split(x, y,test_proportion): # Moro
    x, y = shuffle(x, y, random_state=12)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_proportion)

    return X_train, X_test, Y_train, Y_test

"""we could maybe implement all of this inside the various functions"""

"Takes as variables a classifier, the data and the number of cross-validation that is aimed"
def cross_validation(clf,X_train, train_y,cv_number): #Rodrigues
    cross_validation_values = cross_val_score(clf, X_train, train_y, cv=cv_number)
    print("Cross-Validation Mean Value:",cross_validation_values.mean())

    return cross_validation_values

def decision_tree(X_train, train_y, X_test, test_y): # Gustavo Rodrigues
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,train_y)
    pred_y = clf.predict(X_test)
    print(confusion_matrix(test_y, y_pred))    
    print("Accuracy:",accuracy_score(test_y, pred_y))
    print(classification_report(test_y, y_pred))
    return
