import config
import numpy as np
import pandas as pd


# Hypothesis Function
def hypothesis(thetarg, x):
    return np.dot(x, thetarg)


# Cost function used to calculate the error between hypothetic and actual value over m training examples
def cost(x, y, thetarg, m):
    return float((1 / (2 * m)) * np.dot((hypothesis(thetarg, x) - y).T, (hypothesis(thetarg, x) - y)))


# Normal Equation method to minimize cost function in configurable alpha and iterations
def normal_equation(x, y, m):
    inverse = np.linalg.inv(np.dot(x.T, x))
    thetarg = np.dot(np.dot(inverse, x.T), y)
    cost_value = cost(x, y, thetarg, m)
    return cost_value, thetarg


muldata = pd.read_csv(config.path)
indata = muldata.drop('price', axis=1)
indata.insert(0, "x0", 1)
outdata = muldata['price']

inmatrix = indata.values
outmatrix = outdata.values.reshape(outdata.size, 1)

msize = len(outmatrix)

cost_final, theta_final = normal_equation(inmatrix, outmatrix, msize)
