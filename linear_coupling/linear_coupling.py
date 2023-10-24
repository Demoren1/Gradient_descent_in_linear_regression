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

        self.ordinary_steps = [self.x[1]]
        self.aggressive_steps = [self.x_private[1]]

        self.y_pred = 0
        self.y_pred_private = 0
        
        self.cost_list = []

    def find_gradient(self, momentum):
        d_theta = 1 / self.m * self.x.T.dot(self.y_pred - self.y)
        d_theta_private = 1 / self.m_private * self.x_private.T.dot(self.y_pred_private - self.y_private)
        
        d_theta_private = momentum * d_theta_private + (1 - momentum) * d_theta

        return d_theta, d_theta_private


    def update_theta(self,learning_rate_1, learning_rate_2, momentum):
        self.y_pred = np.dot(self.x, self.theta)
        self.y_pred_private = np.dot(self.x_private, self.theta)

        d_theta, d_theta_private = self.find_gradient(momentum)

        self.ordinary_steps.append(self.theta - learning_rate_1 * d_theta)
        self.aggressive_steps.append(np.array(self.aggressive_steps[-1]).reshape(2, 1) - learning_rate_2 * d_theta_private)

        self.theta = momentum * np.array(self.aggressive_steps[-1]) + (1 - momentum) * np.array(self.ordinary_steps[-1])

        self.old_thetas.append(np.copy(self.theta))


    def compute_cost(self):
        cost = 0
        cost += 1 / (2 * self.m) * np.sum(np.square(self.y_pred - self.y, dtype=np.float64))
        cost += 1 / (2 * self.m_private) * np.sum(np.square(self.y_pred_private - self.y_private, dtype=np.float64))

        self.cost_list.append(cost)
        return cost
    

    def get_current_accuracy(self):
        x_check = [i[1] for i in self.x]
        y_check = [i[0] for i in self.y]

        k, b, *_ = _mnk(x_check, y_check)

        true_theta = np.array([b, k]).reshape(2, 1)
        theta = np.sum(np.array(self.old_thetas), axis=0) / len(self.old_thetas)
        
        accuracy = float(np.linalg.norm(true_theta - theta) / np.linalg.norm(true_theta))
        return accuracy
    

    def show_graphs(self):
        x = self.x
        y = self.y

        x_private = self.x_private
        y_private = self.y_private
        
        theta = np.sum(np.array(self.old_thetas), axis=0) / len(self.old_thetas)
        # print(self.old_thetas)
        # print("theta is\n", theta)

        cost_list = self.cost_list

        tmp_x = list(x[:, 1]) 
        tmp_y = list(y[:, 0]) 

        x_check = [elem for elem in tmp_x]
        y_check = [elem for elem in tmp_y]

        k, b, *_ = _mnk(x_check, y_check)

        x_line = [0, max(x_check)]
        y_line = [k * x_line[0] + b, k * x_line[1] + b]


        print("predictable b =", theta[0], ", predictable k =", theta[1])
        print("b =", b, ",k =", k)

        plt.plot(list(x[:, 1]), y, '.', color="blue", label="data")
        plt.plot(list(x_private[:, 1]), y_private, '.', color="red", label="private data")

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