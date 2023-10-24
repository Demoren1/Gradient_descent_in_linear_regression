import linear_coupling
import numpy as np
import data_manager

def main():
    file_1 = "../datasets/dataset.txt"
    file_2 = "../datasets/private_dataset.txt"
    iteration = 1000000
    learning_rate_1 = 0.0001
    learning_rate_2 = 0.0001
    momentum = 0.5

    # data_manager.show_2_datasets(file_1, file_2)
    
    data = np.loadtxt(file_1, delimiter=' ')
    private_data = np.loadtxt(file_2, delimiter=' ')

    model = linear_coupling.linear_regression_coupling(data, private_data)
    train_model(model, iteration, learning_rate_1, learning_rate_2, momentum)
    model.show_graphs()

    print("accuracy is ", model.get_current_accuracy())


def train_model(model : linear_coupling.linear_regression_coupling, iteration, learning_rate_1, learning_rate_2, momentum):
    for i in range(iteration):
        model.update_theta(learning_rate_1, learning_rate_2, momentum)
        model.compute_cost()
    return

main()

    