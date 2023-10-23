from MNK import *
from math import *
import random

plt.rcParams ['figure.figsize'] = [8, 8]

k = 1
limit = 100
error = 20

X = []
Y = []

with open("dataset.txt", "w") as file:
    for i in range(limit):
        for j in range (5):
            x = i + error * random.random()
            y = k * i + error * random.random()

            line = [str(x)[:5] for x in [x, y]]
            line = " ".join(line)
            line += "\n"
            file.write(line)

            X.append(x)
            Y.append(y)

k, b, k_error, b_error = mnk(X, Y)

print("k = %.5g\nb = %.5g\nk_error = %.5g\nb_error = %.5g " % (k, b, k_error, b_error))

x_error = [0, limit + error]
y_error = [k * x_error[0] + b, k * x_error[1] + b]

fig1, y_x = plt.subplots()
plt.grid()

y_x.set_xlabel(r"$x$")
y_x.set_ylabel(r"$y$")

y_x.plot(X, Y, '.')
y_x.plot(x_error, y_error, '-')

plt.show()



    