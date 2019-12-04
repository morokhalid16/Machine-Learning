import pandas as pd
import numpy as np
from sklearn import preprocessing, decomposition
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier


# If flag is True, we are working with kidney_disease.csv file
# If flag is False, we are working with data_banknote_authentication.txt file

def load_data(path, flag): # Pastorino & Riera i Marin
    if (flag == False):
        column = ["var", "skewness", "curtosis", "entropy", "class"]
        data = pd.read_csv(path, header=None, names=column)
    else:
        data = pd.read_csv(path)
    return data


def visualize_data(data):  # Pastorino & Riera i Marin
    """Visualizes in a Boxplot the data once cleaned"""
    print(data.describe())
    column = data.columns
    sns.boxplot(data = data)
    sns.pairplot(data)
    n = len(column)
    f, ax = plt.subplots(1, n, figsize=(10,3))

    for i in range(n):
        vis = sns.distplot(data[column[i]],bins=10, ax= ax[i])
        f.savefig('subplot.png')


def pca(x_cleaned, n_components):  # Moro
    std_scale = preprocessing.StandardScaler().fit(x_cleaned)
    x_scaled = std_scale.transform(x_cleaned)
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(x_scaled)
    dataset = pca.transform(x_scaled)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())


def svm(x_train, y_train, x_test, y_test, kern):  # Pastorino & Riera i Marin
    svclassifier = SVC(kernel=kern)
    clf = svclassifier.fit(x_train, y_train)
    y_pred = svclassifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    return clf


# tested only on banknote authentication for the moment
# works well with following values n_layers = 3, n_neurons_per_layer = [4,8,1]
# kernel_init = 'uniform', activ = 'relu'
# for the chronic kidney disease -> 2 layers, 256 neurons first, 1 neuron last
# kernel_init = 'uniform', activ = 'relu'


def MLP(n_layers, n_neurons_per_layer, x_train, x_test, y_test, y_train, kernel_init, activ, batch_size, epochs): # Pastorino
    model = Sequential()
    model.add(Dense(n_neurons_per_layer[0], kernel_initializer=kernel_init, activation=activ, input_shape=(x_train.shape[1],)))
    for i in range(1, n_layers-1):
        model.add(Dense(n_neurons_per_layer[i], kernel_initializer=kernel_init))
    model.add(Dense(1, kernel_initializer=kernel_init, activation='sigmoid'))
    model.summary()
    sgd = keras.optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size, epochs)
    np.set_printoptions(precision=4, suppress=True)
    eval_results = model.evaluate(x_test, y_test, verbose=0)
    print("\nLoss, accuracy on test data: ")
    print("%0.4f %0.2f%%" % (eval_results[0], eval_results[1]*100))
    return model


def split(x, y,test_proportion): # Moro
    x, y = shuffle(x, y, random_state=12)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_proportion)

    return x_train, x_test, y_train, y_test


"Takes as variables a classifier, the data and the number of cross-validation that is aimed"
def cross_validation(clf,x_train, y_train,cv_number): # Rodrigues
    cross_validation_values = cross_val_score(clf, x_train, y_train, cv=cv_number)
    print("Cross-Validation Mean Value:",cross_validation_values.mean())

    return cross_validation_values


def decision_tree(x_train, y_train, x_test, y_test, max_d, crit):  # Riera i Marin
    clf = DecisionTreeClassifier(max_depth=max_d, criterion=crit)
    clf = clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:",accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return clf


def cleaning_function(data,flag): # Rodrigues
    if(flag):
        data.classification=data.classification.replace("ckd\t","ckd")
        data.drop("id",axis=1,inplace=True)
        data.classification=[1 if each=="ckd" else 0 for each in data.classification]
        non_numeric_atributes = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
        string_but_numeric = ['pcv','wc', 'rc']
        for element in string_but_numeric:
            #Substitute all NaNs to '0'
            data[element]=['0' if type(each) is not type('string') else each for each in data[element]]

            # Eliminate all \t from the beggining of the strings
            data[element]=[each[1:] if each[0] == '\t' else each for each in data[element]]

            # Substitute all '?' to '0'
            data[element]=['0' if '?' in each else each for each in data[element]]

            # Convert String values to numerical values
            data[element] = pd.to_numeric(data[element])

        # Eliminate all \t from the beggining of the strings
        data['dm']=['0' if type(each) is not type('string') else each for each in data['dm']]
        data['dm']=[each[1:] if each[0] == '\t' else each for each in data['dm']]
        data['dm']=[each[1:] if each[0] == ' ' else each for each in data['dm']]
        most_occurrances = ['yes' if len(data.query('dm == "yes"')['dm']) > len(data.query('dm == "no"')['dm']) else 'no']

        data['dm']=[most_occurrances[0] if each == '0' else each for each in data['dm']]

        data['cad']=['0' if type(each) is not type('string') else each for each in data['cad']]
        data['cad']=[each[1:] if each[0] == '\t' else each for each in data['cad']]
        most_occurrances = ['yes' if len(data.query('cad == "yes"')['cad']) > len(data.query('cad == "no"')['cad']) else 'no']

        data['cad']=[most_occurrances[0] if each == '0' else each for each in data['cad']]

        numeric_attributes = ['age','bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

        #For all the numeric attributes replacing the nan's by the mean of the column
        for attribute in numeric_attributes:
            mean_value = data[attribute].mean()
            data[attribute]=[mean_value if np.isnan(each) else each for each in data[attribute]]

        #Normalizing each numerical variable
        for attribute in numeric_attributes:
            min_value = data[attribute].min()
            max_value = data[attribute].max()
            amplitude = max_value - min_value
            data[attribute]=[(each - min_value)/amplitude for each in data[attribute]]

        numeric_attributes = ['age','bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        x_cleaned = data[numeric_attributes]
        y_cleaned = data['classification']
        return x_cleaned,y_cleaned
    else:
        x_cleaned = data.drop([data.columns[-1]], axis = 1)
        y_cleaned = data[data.columns[-1]]
        return x_cleaned,y_cleaned
