#!/usr/bin/env python3

import numpy as np

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

