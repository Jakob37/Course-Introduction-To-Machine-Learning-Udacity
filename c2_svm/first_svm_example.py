import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

from sklearn.svm import SVC
clf = SVC(kernel="linear")

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

