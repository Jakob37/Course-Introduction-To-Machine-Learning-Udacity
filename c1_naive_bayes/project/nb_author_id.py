#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train - Features for training data
# features_test - Features for test data
# label_test - Labels for test data
# label_train - Labels for train data

# The features here is an array with mostly floats
# The labels here is a list with one or zeroes
# We know the labels. They are one or zero depending on author.
# Features should be some kind of information derived from text. Word
# frequency!

#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print("Training time: {}s".format(round(time()-t0,3)))

pred = clf.predict(features_test)

accuracy = accuracy_score(labels_test, pred)

print("Found accuracy: {}".format(accuracy))

#########################################################


