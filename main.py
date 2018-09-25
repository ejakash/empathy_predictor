import os
import sys

import handle_dataset
from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import pickle
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso, LogisticRegression)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

file_name = 'trained_model'


def get_trained_SVM_model(X, Y, tune_hyper_parameters=True):
    if tune_hyper_parameters:
        Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        gammas = [0.0001, 0.001, 0.01, 0.1, 1, 'auto']
        degrees = np.arange(0, 3, 1).tolist()
        coef0 = np.arange(0.0, 1.0, 0.1).tolist()
        param_grid = [
            {'C': Cs, 'gamma': gammas, 'degree': degrees, 'coef0': coef0, 'kernel': ['poly']},
            {'C': Cs, 'kernel': ['linear']},
            {'C': Cs, 'gamma': gammas, 'kernel': ['rbf']},
        ]
        gs_clf = GridSearchCV(svm.SVC(), param_grid, n_jobs=-1)
        gs_clf.fit(X, Y)
        print(gs_clf.best_score_)
        print(gs_clf.best_params_)
        pickle.dump(gs_clf.best_estimator_, open(file_name, 'wb'))
        return gs_clf.best_estimator_
    else:
        clf = getPrecomputedSVMModel()
        clf.fit(X, Y)
        return clf


def getPrecomputedSVMModel():
    if os.path.isfile(file_name):
        return pickle.load(open(file_name, 'rb'))
    print('Unable to find the saved model file. Using a previous set to create model')
    clf = svm.SVC(C=1, gamma=0.01, kernel='rbf')
    return clf


def compute_accuracy(clf, X, Y):
    Ypred = clf.predict(X)  # predict the training data
    return np.mean(Y == Ypred)  # check to see how often the predictions are right


if __name__ == '__main__':
    tune_hyper_parameters = False

    if len(sys.argv) == 2:
        if sys.argv[1] == 'true':
            tune_hyper_parameters = True
    data = handle_dataset.peopleData
    print('Doing Recursive Feature Elimination')
# Dummy Baseline classifier.
    dummy = DummyClassifier()
    dummy.fit(data.X, data.Y)
# Do recursive Feature Elimination
    rfe = RFE(svm.LinearSVC(), n_features_to_select=120)
    rfe = rfe.fit(data.X, data.Y)
    rfx_Xtr = rfe.transform(data.X)
    rfx_Xte = rfe.transform(data.Xte)

# Do Principle Component Analysis
    print('Doing PCA')
    variance = .90
    pca = PCA(variance)
    pca.fit(rfx_Xtr, data.Y)
    Xtr = pca.transform(rfx_Xtr)
    Xte = pca.transform(rfx_Xte)

    Ytr = data.Y
    Yte = data.Yte

    print('Getting/training the model')
    clf = get_trained_SVM_model(Xtr, Ytr, tune_hyper_parameters)

    print('Evaluating')
    dtrAcc = compute_accuracy(dummy, data.X, data.Y)
    dteAcc = compute_accuracy(dummy, data.Xte, data.Yte)

    trAcc = compute_accuracy(clf, Xtr, Ytr)
    teAcc = compute_accuracy(clf, Xte, Yte)
    print("Baseline Training accuracy {0}, Baseline test accuracy {1}".format(dtrAcc, dteAcc))
    print("Training accuracy {0},  test accuracy {1}".format(trAcc, teAcc))
