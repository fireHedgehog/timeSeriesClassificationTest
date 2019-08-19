# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 10:18:23 2018

@author: rpear
"""

# from sympy import plot
# print("Hello World")

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from pandas.plotting import scatter_matrix
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

path = "Iris.xlsx"
rawdata = pd.read_excel(path)
print("data summary")
print(rawdata.describe())
nrow, ncol = rawdata.shape
print(nrow, ncol)
print("\n correlation Matrix")
print(rawdata.corr())
rawdata.hist()
plt.show()
# display correlations between all pairs of features
scatter_matrix(rawdata, figsize=[5, 5])
plt.show()
# boxplot
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
ax.boxplot(rawdata.values)
ax.set_xticklabels(['Petal Length', 'Petal Width', 'Sepal Length', 'Sepal Width', 'Class'])
plt.show()

# get the predictors – all columns from 0 to last but one
predictors = rawdata.iloc[:, :ncol - 1]
print(predictors)
target = rawdata.iloc[:, -1]  # index to last column to obtain class values
# By referring to http://scikit-learn.sourceforge.net/stable/modules/generated/sklearn.cross_validation
# .train_test_split.html complete the right hand side of the line below and set the training set size to 70% of the
# size of the dataset
pred_train, pred_test, tar_train, tar_test = train_test_split()
split_threshold = 20
for i in range(2, split_threshold):
    # By referring to https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html set
    # the entropy criterion for splitting and set the minimum no of samples (objects) for splitting a decision node
    # to 1
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(pred_train, tar_train)

    predictions = classifier.predict(pred_test)

    print("Accuracy score of our model with Decision Tree:", i, accuracy_score(tar_test, predictions))

# Naive Bayes classification
gnb = GaussianNB()  # suitable for numeric features
gnb.fit(pred_train, np.ravel(tar_train, order='C'))
predictions = gnb.predict(pred_test)

print("Accuracy score of our model with Gaussian Naive Bayes:", accuracy_score(tar_test, predictions))

# By referring suitable information online identify the version of Naive Bayes
# suitable for classifying discrete data and fill in the line below
mnb = ()  # optimized for nominal features but can work for numeric ones as well
mnb.fit(pred_train, np.ravel(tar_train, order='C'))
predictions = mnb.predict(pred_test)

print("Accuracy score of our model with Multinomial Naive Bayes:", accuracy_score(tar_test, predictions))

# kNN classification

# By referring to https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html specify
# the K parameter and the search method to be KDTree (a type of indexing method)that searches for neighbors faster
# than brute force search
nbrs = KNeighborsClassifier()
nbrs.fit(pred_train, np.ravel(tar_train, order='C'))
predictions = nbrs.predict(pred_test)
print("Accuracy score of our model with kNN :", accuracy_score(tar_test, predictions))

# By referring to https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# specify the activation function to be logistic, solver to be stochastic gradient descent, the learning rate to be 0.1
# two hidden layers, containing 5 and 2 neurons respectively.
clf = MLPClassifier()
clf.fit(pred_train, np.ravel(tar_train, order='C'))
print("Accuracy score of our model with MLP :", accuracy_score(tar_test, predictions))
scores = cross_val_score(clf, predictors, target, cv=10)
print("Accuracy score of our model with MLP under cross validation :", scores.mean())
