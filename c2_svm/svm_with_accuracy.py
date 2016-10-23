import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

import time

features_train, labels_train, features_test, labels_test = makeTerrainData()

from sklearn.svm import SVC
clf = SVC(kernel="linear")

t0 = time.time()
clf.fit(features_train, labels_train)
print("Training time: {}".format(time() - t0))

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print("Accuracy is: {}".format(accuracy))
