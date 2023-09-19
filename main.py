#!/usr/bin/env python3 

import os 
import numpy as np
from gp import ZeroGProcess
from optimization import UpperConfidenceBound
from optimization import ExpectedImprovement
from optimization import ShapeTransferBO


if __name__ == "__main__":
    # Test EI
    EI = ExpectedImprovement()
    EI.get_data_from_file("data/experiment_points.tsv")
    print(EI.X)
    print(EI.Y)

    kessi = 15
    x1 = [14, 12]
    print("EI({:.2f}) = {:.2f}".format(x1[0], EI.aux_func_ei(x1, kessi)))
    x2 = [10.4, 4]
    print("EI({:.2f}) = {:.2f}".format(x2[0], EI.aux_func_ei(x2, kessi)))

    print("Grad at ", x2, " ", EI.auto_grad_ei(x2))
    print("Gard at ", x1, " ", EI.auto_grad_ei(x1))

    start_points = [[12, 3], [3, 1], [13, 12], [23, 11]]
    next_point, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=0.5, num_step=10)
    #EI.plot_ei(kessis=[0.0], highlight_point=[next_point, next_point_aux])

    # Test STBO
    STBO = ShapeTransferBO()
    STBO.get_data_from_file("./data/experiment_points_task2.tsv")
    STBO.build_task1_gp("./data/experiment_points_task1.tsv")
    STBO.build_diff_gp()

    print(STBO.X)
    print(STBO.Y)

    print(STBO.zeroGP1.X)
    print(STBO.zeroGP1.Y)

    print(STBO.diffGP.X)
    print(STBO.diffGP.Y)

    print(STBO.compute_mle_sigma2())
    print(STBO.diffGP.compute_mle_sigma2())

    for x_k in range(10, 20, 1):
        print("grad at ", x_k, " ", STBO.auto_grad_ei([x_k]))

    start_points = [[0], [11], [15]]
    next_point, next_point_aux = STBO.find_best_NextPoint_ei(start_points, learn_rate=0.5, num_step=60)    
    STBO.plot_ei(kessis=[0.0], num_points=300, highlight_point=[next_point, next_point_aux])
    print(STBO.aux_func_ei(next_point))
