import time
t=time.time()

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

#Assign column name
columns = ['F' + str(i) for i in range(130)]

#Read data and store in a DataFrame
data = pd.read_csv('../data/data.csv', sep=' ', header=None, names = columns)
target = pd.read_csv('../data/target.csv', header=None)


#Convert DataFrame to an array
X = data.to_numpy()
y = target.to_numpy().flatten()

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Feature Normalization (data standardization) of train and test data using only TRAINING DATA for both Training set and Validation Set,
#to sure our model generalize well on new, unseen data.

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)

#Define a function to train many classifier at once
def valid_classifier(clf, X_train, y_train, X_valid, y_valid):
    clf.fit(X_train, y_train)
    acc_train = clf.score(X_train, y_train)
    acc_valid = clf.score(X_valid, y_valid)
    return acc_train, acc_valid


names = [
    "Linear SVM",
    "SVM Classifier",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes"
]

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(),
    DecisionTreeClassifier(max_depth=5, random_state=0),
    RandomForestClassifier(max_depth=5, n_estimators=100,random_state=0, max_features=1),
    #to avoid overfitting define early_stopping
    MLPClassifier(hidden_layer_sizes=(100), alpha=1, max_iter=1000, random_state=0, early_stopping=True),
    AdaBoostClassifier(),
    GaussianNB()
]

#iterate over classifiers
for name, clf in zip(names, classifiers):
    acc_train, acc_valid = valid_classifier(clf, X_train, y_train, X_valid, y_valid)
    print(name, "Train Accuracy: %.2f%% Validation Accuracy: %.2f%%" % (acc_train * 100.0,acc_valid * 100.0 ))
    
elapsed=time.time()-t
elapsed