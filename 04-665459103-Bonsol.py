#
# CS 559 Neural Networks HW #4
# Author: Kai Bonsol
#

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

np.random.seed(3)

def initialize_w():
    w00 = np.random.uniform(0, 1, 1)
    w01 = np.random.uniform(0, 1-w00, 1)
    return np.array([w00, w01])

# function to optimize
def f(x, y):
    return -np.log(1-x-y) - np.log(x) - np.log(y)

# gradient of f, vector of first partial derivatives
def gradient_f(x, y):
    fx = (1/(1-x-y)) - 1/x
    fy = (1/(1-x-y)) - 1/y
    return np.array([fx, fy])

# hessian of f, matrix of second partial derivatives
def hessian_f(x, y):
    fxx = (1/pow(1-x-y, 2)) + 1/pow(x, 2)
    fyy = (1/pow(1-x-y, 2)) + 1/pow(y, 2)
    fxy = 1/pow(1-x-y, 2)
    H = np.array([[fxx, fxy], [fxy, fyy]])
    H = np.squeeze(H, axis=(2,))
    return H

def gradient_descent(w, learning_rate):
    print("Preforming gradient descent on f(x, y)...")
    print("Initial w: ", w)
    energies = []
    xpoints = []
    ypoints = []
    while True:
        new_w = w - learning_rate * gradient_f(w[0], w[1])
        x = new_w[0]
        y = new_w[1]
        if np.allclose(new_w, w, rtol=1e-03): # convergence condition
            w = new_w
            break
        elif x + y >= 1 or x <= 0 or y <= 0:  # not in valid domain
            learning_rate /= 2
            w = initialize_w()
            continue
        energies.append(f(x, y))
        xpoints.append(x)
        ypoints.append(y)

        w = new_w
    print("Finished with gradient descent.")
    print("W*:", w)
    return (w, energies, xpoints, ypoints)

def newtons_method(w, learning_rate):
    print("Preforming newton's method on f(x, y)...")
    print("Initial w: ", w)
    energies = []
    xpoints = []
    ypoints = []
    beta = 100
    while True:
        g = gradient_f(w[0], w[1])
        H = hessian_f(w[0], w[1])
        new_w = w - learning_rate * np.matmul(np.linalg.inv(H), g) 
        print(new_w)
        x = new_w[0]
        y = new_w[1]
        if np.allclose(new_w, w, rtol=1e-03): # convergence condition
            w = new_w
            break
        elif x + y >= 1 or x <= 0 or y <= 0:  # not in valid domain
            learning_rate /= 2
            w = initialize_w()
            continue
        energies.append(f(x, y))
        xpoints.append(x)
        ypoints.append(y)

        w = new_w
    print("Finished with newton's method.")
    print("W*:", w)
    return (w, energies, xpoints, ypoints)


def plot_energies(w, energies, xpoints, ypoints):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(xpoints, ypoints, energies)
    X, Y = np.meshgrid(np.linspace(-0.1, 1, 50), np.linspace(-0.1, 1, 50))
    Z = f(X, Y)
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    fig.show()

def plot_epoch_energies(energies):
    fig, ax = plt.subplots()
    epochs = []
    for i in range(1, len(energies)+1):
        epochs.append(i)
    ax.plot(epochs, energies)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Energy")
    fig.show()
    
    
w_init = initialize_w()
w, energies, xpoints, ypoints = gradient_descent(w_init, 1)
plot_energies(w, energies, xpoints, ypoints)
plot_epoch_energies(energies)
w, energies, xpoints, ypoints = newtons_method(w_init, 1)
plot_energies(w, energies, xpoints, ypoints)
plot_epoch_energies(energies)
