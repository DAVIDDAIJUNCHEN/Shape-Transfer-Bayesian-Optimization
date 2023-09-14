#!/usr/bin/env python3 

import os 
import numpy as np
from gp import ZeroGProcess
from optimization import UpperConfidenceBound, ExpectedImprovement


if __name__ == "__main__":
    # Test EI
    EI = ExpectedImprovement()
    EI.get_data_from_file("data/experiment_points.tsv")
    print(EI.X)
    print(EI.Y)

    kessi = 15
    x1 = [14]
    print("EI({:.2f}) = {:.2f}".format(x1[0], EI.aux_func(x1, kessi)))
    x2 = [10.4]
    print("EI({:.2f}) = {:.2f}".format(x2[0], EI.aux_func(x2, kessi)))

    # print("grad(18)", EI.auto_grad([18], num_mc=10000))
    # print("grad(18.2)", EI.auto_grad([18.2], num_mc=10000))
    # print("grad(18.4)", EI.auto_grad([18.4], num_mc=10000))
    # print("grad(18.6)", EI.auto_grad([18.6], num_mc=10000))
    # print("grad(18.8)", EI.auto_grad([18.8], num_mc=10000))
    # print("grad(19)", EI.auto_grad([19], num_mc=10000))
    # print("grad(19.3)", EI.auto_grad([19.3], num_mc=10000))
    # print("grad(19.6)", EI.auto_grad([19.6], num_mc=10000))
    # print("grad(20)", EI.auto_grad([20], num_mc=10000))
    # print("grad(20.5)", EI.auto_grad([20.5], num_mc=10000))
    # print("grad(21)", EI.auto_grad([21], num_mc=10000))
    # print("grad(21.5)", EI.auto_grad([21.5], num_mc=10000))
    # print("grad(22)", EI.auto_grad([22], num_mc=10000))

    next_point, next_point_aux = EI.find_best_NextPoint([[12], [3], [13], [23]], learn_rate=0.5, num_step=10)
    EI.plot(kessis=[0.0], highlight_point=[next_point, next_point_aux])
