import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Hypothesis Function
def hypothesis(thetarg, x):
    return np.dot(x, thetarg)


# Cost function used to calculate the error between hypothesis and actual value over m training examples
def cost(x, y, thetarg, m):
    return float((1 / (2 * m)) * np.dot((hypothesis(thetarg, x) - y).T, (hypothesis(thetarg, x) - y)))


# Gradient Descent method to minimize cost function in configurable alpha and iterations
def gradient_descent(x, y, thetarg, m):
    jvec = []
    theta_history = []
    for i in range(config.num_iterations):
        theta_history.append(list(thetarg[:, 0]))
        jvec.append(cost(x, y, thetarg, m))
        for j in range(len(thetarg)):
            thetarg[j] = thetarg[j] - (alpha / m) * np.sum((hypothesis(thetarg, x) - y) *
                                                           np.array(x[:, j]).reshape(m, 1))
    return thetarg, theta_history, jvec


muldata = pd.read_csv(config.path)
indata = muldata.drop('price', axis=1)
indata.insert(0, "x0", 1)
outdata = muldata['price']

normin = (indata - indata.mean()) / indata.std()
normin.fillna(0, inplace=True)
normout = (outdata - outdata.mean()) / outdata.std()

inmatrix = normin.values
outmatrix = normout.values.reshape(outdata.size, 1)

theta = np.zeros([inmatrix.shape[1], 1])
msize = len(outmatrix)
num_iterations = config.num_iterations
alpha = config.alpha

theta_final, theta_hist, compute_cost = gradient_descent(inmatrix, outmatrix, theta, msize)

iterations = list(range(1, config.num_iterations))
compute_cost.pop(0)
plt.title("Cost Function fall with iterations")
plt.xlabel("Number of iterations")
plt.ylabel("Cost Function")
plt.plot(iterations, compute_cost)
