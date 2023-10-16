import numpy as np
import matplotlib.pyplot as plt
from MNK import *

data = np.loadtxt("dataset.txt", delimiter=' ')

x = data[:, 0]
y = data[:, 1]
y = y.reshape(y.size, 1)

x = np.vstack((np.ones((x.size, )), x)).T

def model(x, y, learning_rate, iteration):
    m = y.size
    theta = np.zeros((2, 1))
    cost_list = []
    
    for i in range(iteration):
        y_pred = np.dot(x, theta)
        cost = 1 / (2 * m) * np.sum(np.square(y_pred - y, dtype=np.float64))

        d_theta = 1 / m * x.T.dot(y_pred - y)
        theta = theta - learning_rate * d_theta

        cost_list.append(cost)

    return theta, cost_list

def train_model():
    iteration = 1000
    learning_rate = 0.00001

    theta, cost_list = model(x, y, learning_rate, iteration)

    check_model(theta, x, y, cost_list)
    return 


def check_model(theta, x, y, cost_list):
    x_check = [i[1] for i in x]
    y_check = [i[0] for i in y]

    k, b, k_error, b_error = mnk(x_check, y_check)

    x_line = [0, x_check[-1]]
    y_line = [k * x_line[0] + b, k * x_line[1] + b]

    tmp_theta = [i[0] for i in theta]
    print(tmp_theta)
    print(k, b)

    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], np.dot(x, theta), color='red', label="Predictable line")
    plt.plot(x_line, y_line, color='green', label="With MLS")
    plt.legend()
    plt.show()

    rng = [x for x in range(len(cost_list))]
    plt.plot(cost_list, rng)
    plt.plot(0,0)
    plt.grid()
    plt.show()

    return 


train_model()