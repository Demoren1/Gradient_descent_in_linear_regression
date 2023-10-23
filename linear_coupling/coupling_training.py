import linear_coupling
import numpy as np

def main():
    file = "../dataset.txt"
    iteration = 100
    learning_rate_1 = 0.0001
    learning_rate_2 = 0.0001
    momentum = 0.5
    
    
    data = np.loadtxt(file, delimiter=' ')
    print(type(data))
    model = linear_coupling.Linear_Regression_Coupling(data)

    train_model(model, iteration, learning_rate_1, learning_rate_2, momentum)
    print("accuracy is ", model.get_current_accuracy())
    model.show_graphs()


def train_model(model : linear_coupling.Linear_Regression_Coupling, iteration, learning_rate_1, learning_rate_2, momentum):
    for i in range(iteration):
        model.update_theta(learning_rate_1, learning_rate_2, momentum)
        model.compute_cost()
    return

main()

    