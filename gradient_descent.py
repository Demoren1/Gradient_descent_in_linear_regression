import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("dataset.txt", delimiter=' ')

x = data[:, 0]
y = data[:, 1]
y = y.reshape(y.size, 1)

x = np.vstack((np.ones((x.size, )), x)).T

# print(x.shape)
# print(y.shape)

def model(x, y, learning_rate, iteration):
    m = y.size
    theta = np.zeros((2, 1))
    
    for i in range(iteration):
        y_pred = np.dot(x, theta)
        cost = 1 / (2 * m) * np.sum(np.square(y_pred - y))

        d_theta = 1 / m * x.T.dot(y_pred - y)
        theta = theta - learning_rate * d_theta

    return theta

def train_model():
    iteration = 2000
    learning_rate = 0.000001

    theta = model(x, y, learning_rate, iteration)

    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], np.dot(x, theta), color='red')
    plt.show()

train_model()
