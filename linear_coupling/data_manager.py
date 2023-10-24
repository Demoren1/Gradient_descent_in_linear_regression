import numpy as np
from MNK import *
from math import *
import random

def main():
    file_1 = "datasets/dataset.txt"
    file_2 = "datasets/private_dataset.txt"
    # create_dataset(file_2, 1)
    # show_2_datasets(file_1, file_2)


def create_dataset(file, private_dataset_flag = False):
    plt.rcParams ['figure.figsize'] = [8, 8]

    k = 1
    start = 30
    limit = 20
    error = 5
    density = 10

    X = []
    Y = []

    with open(file, "w") as tmp_file:
        for i in range(start, start + limit):
            for j in range (density):
                tmp_error = error * random.random()
                x = i + error * random.random()
                if (private_dataset_flag and tmp_error < error / 2):
                    continue

                y = x + 7 * tmp_error

                line = [str(x)[:5] for x in [x, y]]
                line = " ".join(line)
                line += "\n"
                tmp_file.write(line)

                X.append(x)
                Y.append(y)

    k, b, k_error, b_error = mnk(X, Y)

    print("k = %.5g\nb = %.5g\nk_error = %.5g\nb_error = %.5g " % (k, b, k_error, b_error))

    x_error = [start, start + limit + error]
    y_error = [k * x_error[0] + b, k * x_error[1] + b]

    fig1, y_x = plt.subplots()
    plt.grid()

    y_x.set_xlabel(r"$x$")
    y_x.set_ylabel(r"$y$")

    y_x.plot(X, Y, '.')
    y_x.plot(x_error, y_error, '-')

    plt.show()

def show_2_datasets(file1, file2):
    data1 = np.loadtxt(file1, delimiter=' ')
    data2 = np.loadtxt(file2, delimiter=' ')

    x1 = data1[:, 0]
    y1 = data1[:, 1]

    x2 = data2[:, 0]
    y2 = data2[:, 1]

    plt.rcParams ['figure.figsize'] = [8, 8]

    plt.grid()
    plt.plot(x1, y1, '.', label="first dataset", color="blue")
    plt.plot(x2, y2, '.', label="second dataset", color="red")

    plt.legend()
    plt.show()


main()


    


    