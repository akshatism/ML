import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/akshatkumar/Downloads/machine-learning-ex1/ex1/ex1data1.txt")

X = data['population']
Y = data['profit']

X = X.values.reshape(len(X), 1)
Y = Y.values.reshape(len(Y), 1)

X_train = X[:67]
Y_train = Y[:67]

plt.scatter(X_train, Y_train)
plt.title('Linear Regression Train Data')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

X_test = X[67:]
Y_test = Y[67:]

plt.scatter(X_test, Y_test)
plt.title('Linear Regression Test Data')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

regression = linear_model.LinearRegression()
regression.fit(X_test, Y_test)

plt.plot(X_test, regression.predict(X_test))
