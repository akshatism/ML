# Regularized Logistic Regression
import config
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


# Hypothesis Function (Sigmoid)
def hypothesis(thetarg, x):
    return 1 / (1 + np.exp(-np.dot(x, thetarg)))


# Cost function used to calculate the error between hypothetic and actual value over m training examples
def cost(x, y, thetarg, msize, reg):
    return float((1 / msize) * (
            -np.dot(y.T, np.log(hypothesis(thetarg, x))) - np.dot((1 - y).T, np.log(1 - hypothesis(thetarg, x))))) \
           + float((reg / (2 * m)) * np.dot(theta[1:].T, theta[1:]))


# Gradient Descent method to minimize cost function in configurable alpha and iterations
def gradient_descent(x, y, thetarg, msize, reg):
    jvec = []
    theta_history = []
    for i in range(config.num_iterations):
        theta_history.append(list(thetarg[:, 0]))
        jvec.append(cost(x, y, thetarg, msize, reg))

        thetarg[0] = thetarg[0] - (config.alpha / msize) * np.sum((hypothesis(thetarg, x) - y) *
                                                                  np.array(x[:, 0]).reshape(msize, 1))
        for j in range(1, len(thetarg)):
            thetarg[j] = thetarg[j] - config.alpha * (
                    (1 / msize) * np.sum((hypothesis(thetarg, x) - y) * np.array(x[:, j]).reshape(msize, 1)) +
                    (reg / msize) * thetarg[j])
    return thetarg


# Prediction to classify cancer as benign or malignant
def predict(theta_final, input_matrix):
    h = hypothesis(theta_final, input_matrix)
    pred = []
    for i in h:
        if i > 0.5:
            pred.append(1)
        else:
            pred.append(0)
    return pred


# Accuracy check on the training dataset
def accuracy(prediction, decision):
    count = 0
    for i, j in zip(prediction, decision):
        if i == j:
            count = count + 1
    return 100 * (count / len(prediction))


# Loading test1 and test2 as features for training data with decision label
data = np.loadtxt(config.path, delimiter=',', usecols=[0, 1, 2], unpack=True)
X = np.transpose(np.array(data[:-1]))

# Converts input feature to 28 features using PolynomialFeatures
poly = PolynomialFeatures(6)
XX = poly.fit_transform(X)
inmatrix = np.insert(X, 0, 1, axis=1)
outmatrix = np.transpose(np.array(data[-1:]))
m = outmatrix.size
theta = np.zeros([XX.shape[1], 1])

# Getting best parameters by running gradient descent
final_theta = gradient_descent(XX, outmatrix, theta, m, config.reg)

# Prediction and accuracy of the classifier
predict_label = predict(final_theta, XX)
accuracy = accuracy(predict_label, outmatrix)
