import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


def preprocess(file):
    """ 
    Returns a 4-tuple of DataFrames. The first element is the X vector to be used for 
    training, the second the X vector to be used for testing, the third
    the y values for training, the fourth 
    """
    data = pd.read_csv(file, low_memory=False, index_col=0).dropna()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return train_test_split(X,y,random_state=1)


def build_sklearn_nnet_classifier(file):
    """
    Returns an sklearn MLPClassifier trained on the file name input as a parameter and the accuracy score in a tuple.
    

    """
    # Split X and y into training and testing data
    X_train, X_test, y_train, y_test = preprocess(file)
    # MLPs are sensitive to feature scaling, so we scale the data
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # Use same scaling on test data
    X_test = scaler.transform(X_test)

    # Use default parameters
    clf = MLPClassifier()
    # Train the NN on the training data
    clf.fit(X_train, y_train)

    return (clf, clf.score(X_test, y_test))

def build_sklearn_randforest_classifier(file):
    """
    Returns an sklearn RandomForestClassifier trained on the input file and the accuracy score in a tuple.
    """
    X_train, X_test, y_train, y_test = preprocess(file)

    clf = RandomForestClassifier()

    clf.fit(X_train, y_train)

    return (clf, clf.score(X_test, y_test))

def build_sklearn_nnet_regressor(file):
    """
    Returns an sklearn MLPRegressor trained on the input file and the accuracy score in a tuple.
    """
    X_train, X_test, y_train, y_test = preprocess(file)

    clf = MLPRegressor()

    clf.fit(X_train, y_train)

    return (clf, clf.score(X_test, y_test))


def build_sklearn_randforest_regressor(file):
    """
    Returns an sklearn RandomForestRegressor trained on the input file and the accuracy score in a tuple.
    """

    X_train, X_test, y_train, y_test = preprocess(file)

    clf = RandomForestRegressor()

    clf.fit(X_train, y_train)

    return (clf, clf.score(X_test, y_test))


