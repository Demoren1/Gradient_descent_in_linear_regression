import numpy as np
import matplotlib.pyplot as plt
import math

class Linear_Regression:
    def __init__(self, data : np.ndarray):
        self.x = data[:, 0]
        self.y = data[:, 1]

        self.y = self.y.reshape(self.y.size, 1)
        self.x = np.vstack((np.ones((self.x.size, )), self.x)).T
        
        self.cost_list = []
        self.m = self.y.size
        self.theta = np.zeros((2, 1))
        self.y_pred = 0


    def update_theta(self,learning_rate):
        self.y_pred = np.dot(self.x, self.theta)
        d_theta = 1 / self.m * self.x.T.dot(self.y_pred - self.y)
        self.theta = self.theta - learning_rate * d_theta


    def compute_cost(self):
        cost = 0
        cost = 1 / (2 * self.m) * np.sum(np.square(self.y_pred - self.y, dtype=np.float64))

        self.cost_list.append(cost)
        return cost
    

    def get_current_accuracy(self):
        x_check = [i[1] for i in self.x]
        y_check = [i[0] for i in self.y]

        k, *_ = _mnk(x_check, y_check)

        accuracy = float(1 - abs((self.theta[1] - k) / k))
        
        return accuracy
    

    def show_graphs(self):
        x = self.x
        y = self.y
        theta = self.theta
        cost_list = self.cost_list

        x_check = [i[1] for i in x]
        y_check = [i[0] for i in y]

        k, b, k_error, b_error = _mnk(x_check, y_check)

        x_line = [0, x_check[-1]]
        y_line = [k * x_line[0] + b, k * x_line[1] + b]

        tmp_theta = [i[0] for i in theta]
        print(theta)
        print("b =", tmp_theta[0], ",k =", tmp_theta[1])
        print("b =", b, ",k =", k)

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