import config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Hypothesis Function (Sigmoid)
def hypothesis(thetarg, x):
    return 1 / (1 + np.exp(-np.dot(x, thetarg)))


# Cost function used to calculate the error between hypothetic and actual value over m training examples
def cost(x, y, thetarg, msize):
    return float((1 / msize) * (
                -np.dot(y.T, np.log(hypothesis(thetarg, x))) - np.dot((1 - y).T, np.log(1 - hypothesis(thetarg, x)))))


# Gradient Descent method to minimize cost function in configurable alpha and iterations
def gradient_descent(x, y, thetarg, msize):
    jvec = []
    theta_history = []
    for i in range(config.num_iterations):
        theta_history.append(list(thetarg[:, 0]))
        jvec.append(cost(x, y, thetarg, msize))

        for j in range(len(thetarg)):
            thetarg[j] = thetarg[j] - (config.alpha / msize) * np.sum(
                (hypothesis(thetarg, x) - y) * np.array(x[:, j]).reshape(msize, 1))
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
muldata = pd.read_csv(config.path)
data = np.loadtxt(config.path, delimiter=',', usecols=[0, 1, 2], unpack=True)
X = np.transpose(np.array(data[:-1]))
inmatrix = np.insert(X, 0, 1, axis=1)
outmatrix = np.transpose(np.array(data[-1:]))
m = outmatrix.size
theta = np.zeros([inmatrix.shape[1], 1])

# Scatter Plot to visualize positive and negative examples
plt.title("Training Classification")
plt.xlabel("score1")
plt.ylabel("score2")
color = ['red' if l == 0 else 'green' for l in outmatrix]
plt.scatter(inmatrix[:, 1], inmatrix[:, 2], color=color)

# Getting best parameters by running gradient descent
final_theta = gradient_descent(inmatrix, outmatrix, theta, m)

# Prediction and accuracy of the classifier
predict_label = predict(final_theta, inmatrix)
accuracy = accuracy(predict_label, outmatrix)

# Scatter Plot to visualize predicted data
plt.title("Hypothesis Classification")
plt.xlabel("Score1")
plt.ylabel("Score2")
color = ['red' if l == 0 else 'green' for l in predict_label]
plt.scatter(inmatrix[:, 1], inmatrix[:, 2], color=color)
