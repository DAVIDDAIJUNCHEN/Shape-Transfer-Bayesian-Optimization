#!/usr/bin/env python3 

import os 
import numpy as np
from gp import ZeroGProcess
from optimization import UpperConfidenceBound
from optimization import ExpectedImprovement
from optimization import ShapeTransferBO


if __name__ == "__main__":
    # EI on Experiment 1
    EI = ExpectedImprovement()
    EI.get_data_from_file("data/experiment_points_task1.tsv")
    print(EI.X)
    print(EI.Y)

    kessi = 1
    x1 = [12]
    print("EI({:.2f}) = {:.2f}".format(x1[0], EI.aux_func_ei(x1, kessi)))
    x2 = [10.4]
    print("EI({:.2f}) = {:.2f}".format(x2[0], EI.aux_func_ei(x2, kessi)))

    start_points = [[12], [1], [13], [11]]
    next_point, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=0.5, num_step=10)
    EI.plot_ei(kessis=[0.0], num_points=300, highlight_point=[next_point, next_point_aux])

    # STBO on Experiments 1 & 2 
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

    start_points = [[0], [13], [15], [20]]
    next_point, next_point_aux = STBO.find_best_NextPoint_ei(start_points, learn_rate=0.5, num_step=60)    
    STBO.plot_ei(kessis=[0.0], num_points=300, highlight_point=[next_point, next_point_aux])
    print(next_point)
