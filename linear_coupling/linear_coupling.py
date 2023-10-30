import numpy as np
import matplotlib.pyplot as plt
import math

class linear_regression_coupling:
    def __init__(self, data : np.ndarray, private_data : np.ndarray):
        self.x = data[:, 0]
        self.y = data[:, 1]

        self.x_private = private_data[:, 0]
        self.y_private = private_data[:, 1]

        self.y = self.y.reshape(self.y.size, 1)
        self.y_private = self.y_private.reshape(self.y_private.size, 1)

        self.x = np.vstack((np.ones((self.x.size, )), self.x)).T
        self.x_private = np.vstack((np.ones((self.x_private.size, )), self.x_private)).T

        self.m = self.y.size
        self.m_private = self.y_private.size

        self.theta = np.zeros((2, 1))
        self.old_thetas = []
        self.old_thetas.append(np.copy(self.theta))

        self.ordinary_steps = [np.zeros((2, 1))]
        self.aggressive_steps = [np.zeros((2, 1))]

        self.y_pred = 0
        self.y_pred_private = 0
        
        self.cost_list = []

    def find_gradient(self):
        d_theta = 1 / (self.m + self.m_private) * (self.x.T.dot(self.y_pred - self.y) +
                                                   self.x_private.T.dot(self.y_pred_private - self.y_private))
        return d_theta


    def update_theta(self,learning_rate_1, learning_rate_2, momentum):
        self.y_pred = np.dot(self.x, self.theta)
        self.y_pred_private = np.dot(self.x_private, self.theta)

        d_theta= self.find_gradient()

        self.ordinary_steps.append(self.theta - learning_rate_1 * d_theta)
        self.aggressive_steps.append(np.array(self.aggressive_steps[-1]).reshape(2, 1) - learning_rate_2 * d_theta)


        self.theta = momentum * self.aggressive_steps[-1] + (1 - momentum) * self.ordinary_steps[-1]
        self.old_thetas.append(np.copy(self.theta))


    def compute_cost(self):
        cost = 0
        cost += 1 / (2 * self.m + 2 * self.m_private) * (np.sum(np.square(self.y_pred - self.y, dtype=np.float64)) + 
                                                         np.sum(np.square(self.y_pred_private - self.y_private, dtype=np.float64)))

        self.cost_list.append(cost)
        return cost
    

    def get_current_accuracy(self):
        x_check = [i[1] for i in self.x]
        y_check = [i[0] for i in self.y]

        k, *_ = _mnk(x_check, y_check)
        
        theta = np.sum(np.array(self.old_thetas), axis=0) / len(self.old_thetas)
        accuracy = float(1 - abs((k - theta[1]) / k))
        return accuracy
    

    def show_graphs(self, momentum):
        x = self.x
        y = self.y

        x_private = self.x_private
        y_private = self.y_private
        
        theta = np.sum(np.array(self.old_thetas), axis=0) / len(self.old_thetas)

        cost_list = self.cost_list

        tmp_x = list(x[:, 1])
        tmp_x_private = list(x_private[:, 1])

        tmp_y = list(y[:, 0])
        tmp_y_private = list(y_private[:, 0])


        x_check = [elem for elem in tmp_x]
        x_check_private = [elem for elem in tmp_x_private]

        y_check = [elem for elem in tmp_y]
        y_check_private = [elem for elem in tmp_y_private]

        k, b, *_ = _mnk(x_check, y_check)
        k_private, b_private, *_ = _mnk(x_check_private, y_check_private)

        x_line = [0, max(max(x_check), max(x_check_private))]

        k = momentum * k_private + (1 - momentum) * k
        b = momentum * b_private + (1 - momentum) * b

        y_line = [k * x_line[0] + b, k * x_line[1] + b]


        print("predictable b =", theta[0], ", predictable k =", theta[1])
        print("b =", b, ", k =", k)

        plt.grid()

        plt.plot(list(x[:, 1]), y, 'v', color="blue", label="data1")
        plt.plot(list(x_private[:, 1]), y_private, '^', color="red", label="data2")

        tmp_x = np.array(list(x) + list(x_private))

        plt.plot(tmp_x[:, 1], np.dot(tmp_x, theta), color='red', label="Predictable line")
        plt.plot(x_line, y_line, color='green', label="With MLS")
        plt.legend()
        plt.show()

        rng = [x for x in range(len(cost_list))]
        plt.plot(cost_list, rng)
        plt.plot(0,0)
        plt.grid()
        plt.show()

        return 
    

    
def _mnk(xs, ys, showTable=False):
    count = len(xs)
    mx = sum(xs)/count
    x2 = list(map( lambda x: x*x, xs))
    mx2 = sum(x2)/count
    my = sum(ys)/count
    y2 = list(map( lambda x: x*x, ys))
    my2 = sum(y2)/count
    xy = list(map(lambda x, y: x*y, xs, ys))
    mxy = sum(xy)/count
    k = (mxy - mx*my)/(mx2 - mx*mx)
    b = my - k*mx
    sgm_k = 1/math.sqrt(count)*math.sqrt((my2-my*my)/(mx2-mx*mx)-k*k)
    sgm_b = sgm_k*math.sqrt(mx2-mx*mx)
    return k, b, sgm_k, sgm_b