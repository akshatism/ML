import config
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

# Load data using read_csv method
data = pd.read_csv(config.path)

X = data['population']
Y = data['profit']

X = X.values.reshape(len(X), 1)
Y = Y.values.reshape(len(Y), 1)

# Divide into training and testing data
X_train = X[:config.train_data]
Y_train = Y[:config.train_data]

# Scatter Plot of training data Population vs Profit
plt.scatter(X_train, Y_train)
plt.title('Linear Regression Train Data')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

X_test = X[config.test_data:]
Y_test = Y[config.test_data:]

# Scatter Plot of testing data Population vs Profit
plt.scatter(X_test, Y_test)
plt.title('Linear Regression Test Data')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

# import LinearRegression() from sklearn
regression = linear_model.LinearRegression()
regression.fit(X_test, Y_test)

# Predicting and plotting test data
plt.plot(X_test, regression.predict(X_test))
