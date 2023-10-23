import linear_regression
import numpy as np

def main():
    file = "../dataset.txt"
    iteration = 40
    learning_rate = 0.0001
    
    data = np.loadtxt(file, delimiter=' ')
    print(type(data))
    model = linear_regression.Linear_Regression(data)

    train_model(model, iteration, learning_rate)
    print("accuracy is ", model.get_current_accuracy())
    model.show_graphs()

def train_model(model : linear_regression.Linear_Regression, iteration, learning_rate):
    for i in range(iteration):
        model.update_theta(learning_rate)
        model.compute_cost()
    return

main()

    