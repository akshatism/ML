import numpy as np


# Hypothesis Function
def hypothesis(thetarg, x):
    return np.dot(x, thetarg)


# Cost function used to calculate the error between hypothetic and actual value over m training examples
def cost(x, y, thetarg, msize):
    return float((1/(2 * msize)) * np.dot((hypothesis(thetarg, x) - y).T, (hypothesis(thetarg, x) - y)))


# Gradient Descent method to minimize cost function in configurable alpha and iterations
def gradient_descent(x, y, thetarg, msize):

    jvec = []
    thetahistory = []
    for i in range(numIterations):
        thetahistory.append(list(thetarg[:, 0]))
        jvec.append(cost(x, y, thetarg, msize))
        for j in range(len(thetarg)):
            thetarg[j] = thetarg[j] - (alpha/msize) * np.sum((hypothesis(thetarg, x) - y) *
                                                             np.array(X.T[0]).reshape(m, 1))
    return thetarg, thetahistory, jvec


# Loading file using numpy method loadtxt
data = np.loadtxt("/Users/akshatkumar/Downloads/machine-learning-ex1/ex1/ex1data1.txt",
                  delimiter=',', usecols=[0, 1], unpack=True)
# Converting input to m x n dimensional matrix where m are number of training examples and n are features
X = np.transpose(np.array(data[:-1]))
X = np.insert(X, 0, 1, axis=1)
# Converting output to n dimensional vector
Y = np.transpose(np.array(data[-1:]))
m = Y.size
theta = np.zeros([X.shape[1], 1])

# Setting number of iteration for cost function to attain saturation value and alpha to decide the rate to reach optimum
numIterations = 100
alpha = 0.01

thetafinal, thetahistory, jvec = gradient_descent(X, Y, theta, m)

