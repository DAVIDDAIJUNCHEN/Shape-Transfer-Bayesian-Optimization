#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

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

    return branin(input) + 20*x1 - 30*x2

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


if __name__ == "__main__":
    x_draw = np.linspace(0, 7, 100)

    y_theta_1 = [diff_mu1_mu2([0], [ele], theta=1) for ele in x_draw]
    y_theta_1_2 = [diff_mu1_mu2([0], [ele], theta=0.5) for ele in x_draw]
    y_theta_3 = [diff_mu1_mu2([0], [ele], theta=3) for ele in x_draw]
    
    fig, ax = plt.subplots(1, 1)
    ax.set_title("RHKS Norm of mu1-mu2")
    ax.plot(x_draw, y_theta_1_2, label="theta=0.5")
    ax.plot(x_draw, y_theta_1, label="theta=1")
    ax.plot(x_draw, y_theta_3, label="theta=3")

    ax.legend()
    fig.tight_layout()
    fig.savefig("./rhks_norm_mu1_mu2")
