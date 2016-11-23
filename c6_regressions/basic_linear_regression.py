#!/usr/bin/python3

import numpy
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from class_vis import prettyPicture, output_image

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = agesNetWorthData()

def studentReg(ages_train, net_worths_train):

    from sklearn import linear_model
    
    reg = linear_model.LinearRegression()
    reg.fit(ages_train, net_worths_train)

    return reg


reg = studentReg(ages_train, net_worths_train)

plt.clf()
plt.scatter(ages_train, net_worths_train, color='b', label='train data')
plt.scatter(ages_test, net_worths_test, color='r', label='test data')
plt.plot(ages_test, reg.predict(ages_test), color='black')
plt.legend(loc=2)
plt.xlabel('ages')
plt.ylabel('net worths')

plt.savefig('test.png')
output_image('test.png', 'png', open('test.png', 'rb').read())

print('Net worth: {}'.format(reg.predict(27)[0][0]))
print('Slope: {}'.format(reg.coef_[0][0]))
print('Intercept: {}'.format(reg.intercept_[0]))
print('Test score: {}'.format(reg.score(ages_train, net_worths_train)))
print('Train score: {}'.format(reg.score(ages_test, net_worths_test)))




