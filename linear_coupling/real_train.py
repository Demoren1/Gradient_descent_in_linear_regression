import linear_coupling
import numpy as np
import data_manager
from sklearn.datasets import load_svmlight_file
import pandas as pd
import matplotlib.pyplot as plt

def main():
    file_1 = "datasets/breast_cancer.svm"

    iteration = 10000000
    learning_rate_1 = 0.00001
    learning_rate_2 = 0.00001
    momentum = 0.5

 
    features = 10

    general_data = []


    with open(file_1) as file:
        for line in file:
            splitted = line.split()

            if (len(splitted) < 1):
                break
            
            formatted_data = [0 for x in range(features + 1)]
            formatted_data[0] = float(splitted[0]) - 3

            counter = 1
            for i in range(1, features):
        
                if (counter > len(splitted) - 1):
                    break

                word = splitted[counter].split(':')
                if (float(word[0]) != i):
                    formatted_data[i] = 0
                    continue

                formatted_data[int(word[0])] = float(word[1])
                counter += 1

            general_data.append(formatted_data)


    data1 = []
    data2 = []

    feature1 = 2
    feature2 = 3

    for vector in general_data:
        if (vector[0] == 1):
            data1.append((vector[feature1], vector[feature2]))
        else:
            data2.append((vector[feature1], vector[feature2]))
    
    # data_manager.show_2_datasets_dd(data1, data2)

    data1 = set(data1)
    data2 = set(data2)

    data1 = [list(x) for x in data1]        #strange thing
    data2 = [list(x) for x in data2]        #strange thing

    data1 = np.array(data1)
    data2 = np.array(data2)

    
    model = linear_coupling.linear_regression_coupling(data1, data2)
    train_model(model, iteration, learning_rate_1, learning_rate_2, momentum)
    model.show_graphs(momentum)

    print("accuracy is ", model.get_current_accuracy())


def train_model(model : linear_coupling.linear_regression_coupling, iteration, learning_rate_1, learning_rate_2, momentum):
    for i in range(iteration):
        model.update_theta(learning_rate_1, learning_rate_2, momentum)
        model.compute_cost()
    return

main()
