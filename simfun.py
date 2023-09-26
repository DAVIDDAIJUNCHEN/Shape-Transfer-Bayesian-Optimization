#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def branin(input=[3., 4.]):
    "Branin function: "
    assert(len(input) == 2)
    x1 = input[0]
    x2 = input[1]

    branin1 = x2 - 5.1*(x1**2) / (4*np.pi**2) + 5*x1 / np.pi - 6
    branin2 = 10*(1 - 1/(8*np.pi))*np.cos(x1)
    branin = branin1**2 + branin2 + 10

    return branin

def mod_branin(input=[3., 4.]):
    "Modified Branin function: branin(x1, x2) + 20*x1 - 30*x2"
    assert(len(input) == 2)
    x1 = input[0]
    x2 = input[1]

    #mod_bran = branin(input) + 20*x1 - 30*x2
    mod_bran = branin(input) + 20*x1*np.sin(x2)
    
    return mod_bran

def show_branin(x_low=-1, x_up=1, y_low=-1, y_up=1, x_nums=100, y_nums=100):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    X = np.arange(-5, 10, 0.25)
    Y = np.arange(0, 15, 0.25)
    X, Y = np.meshgrid(X, Y)

    branin1 = Y - 5.1*(X**2) / (4*np.pi**2) + 5*X / np.pi - 6
    branin2 = 10*(1 - 1/(8*np.pi))*np.cos(X)
    branin = branin1**2 + branin2 + 10
    Z = branin

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_zlim(-20, 300)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
    ax.set_zlim(-20, 300)

    plt.show()

    fig.savefig("./branin_3d")

    return 0

def exp_mu(input, mu, theta=1):
    """
    Exponential function on ||input - mu||^2,
    mu and x are lists with same length
    """
    assert(len(input) == len(mu))
    square_mu_input = [(xi - mu_i)**2 for xi, mu_i in zip(input, mu)]
    norm2 = sum(square_mu_input)

    exp_mu = np.exp(-0.5 * norm2 / theta**2)

    return exp_mu

def diff_mu1_mu2(mu1, mu2, theta=1):
    "The RKHS norm of exp_mu(,mu1) - exp_mu(,mu2)"
    norm_rkhs = np.sqrt(2 - 2 * exp_mu(mu1, mu2, theta))

    return norm_rkhs

def show_exp(x_low=-1, x_up=1, y_low=-1, y_up=1, theta=1, x_nums=100, y_nums=100):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)

    exp1_norm2 = X**2 + Y**2  
    exp2_norm2 = (X+1.5)**2 + (Y+1.5)**2
 
    exp1 = np.exp(-0.5 * exp1_norm2 / theta**2)
    exp2 = np.exp(-0.5 * exp2_norm2 / theta**2)
    Z1 = exp1
    Z2 = exp2

    surf1 = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    surf2 = ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    ax.set_zlim(0, 1.4)
    fig.colorbar(surf2, shrink=0.5, aspect=10)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_wireframe(X, Y, Z1, rstride=5, cstride=5)
    ax.plot_wireframe(X, Y, Z2, rstride=5, cstride=5)
    ax.set_zlim(0, 1.4)
    
    plt.show()

    fig.savefig("./exp_3d")

    return 0

def plot_rkhs_norm(low_diff=0, up_diff=7, theta_lst=[0.25, 0.5, 1, 1.5, 3]):
    """
    x-axis: mu1 - mu2, 1-dim scenario
    y-axis: diff_mu1_mu2
    """
    x_draw = np.linspace(low_diff, up_diff, 100)
    
    fig, ax = plt.subplots(1, 1)
    ax.set_title("RKHS Norm of mu1-mu2")

    for theta in theta_lst:
        y_theta = [diff_mu1_mu2([0], [ele], theta) for ele in x_draw]
        ax.plot(x_draw, y_theta, label="theta="+str(theta))

    ax.legend()
    fig.tight_layout()
    fig.savefig("./rkhs_norm_mu1_mu2")

    return 0


if __name__ == "__main__":
    plot_rkhs_norm()
    show_branin()
    show_exp()
