#!/usr/bin/env python3 

import os 
import numpy as np
from gp import ZeroGProcess
from optimization import UpperConfidenceBound
from optimization import ExpectedImprovement


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

    start_points = [[12], [3], [13], [23]]
    next_point, next_point_aux = EI.find_best_NextPoint(start_points, learn_rate=0.5, num_step=10)
    EI.plot(kessis=[0.0], highlight_point=[next_point, next_point_aux])
