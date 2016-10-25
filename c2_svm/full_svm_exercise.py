#!/usr/bin/python

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

import sklearn.svm
import sklearn.metrics

# Get preprocessed data
features_train, features_test, labels_train, labels_test = preprocess()

# Prepare test parameter range (here: parameter C)
param_c_values = [1, 10, 100, 1000, 10000]

for c_value in param_c_values:

    clf = sklearn.svm.SVC(kernel="rbf", C=c_value)
    # clf = SVC(kernel="linear")

    t0 = time()
    clf.fit(features_train, labels_train)
    print("Training time: {}".format(time() - t0))

    pred = clf.predict(features_test, labels_test)

    accuracy = sklearn.metrics.accuracy_score(labels_test, pred)

    print("Accuracy: {}".format(accuracy))




