#!/usr/bin/env /mnt/users/daijun_chen/tools/miniconda3.10/install/envs/python3_huggingface/bin/python3

import os 
import numpy as np

from gp import ZeroGProcess
from optimization import UpperConfidenceBound
from optimization import ExpectedImprovement
from optimization import ShapeTransferBO
from optimization import BiasCorrectedBO


if __name__ == "__main__":
    """" Run Real experiment """
    task = 1

    if task == 1:
        # Part 1: EI on Experiment 1
        # 1.1 ZeroGP EI initialization
        EI_task1 = ExpectedImprovement()
        EI_task1.get_data_from_file("data/experiment_points_task1_gp.tsv")
        print(EI_task1.X)
        print(EI_task1.Y)

        # 1.2 AC optimization 
        # dim = EI.dim
        # low_opt1 = 0, high_opt1 = 1000
        # num_start_opt1 = 15
        # start_points = [np.random.uniform(low_opt1, high_opt1, size=dim).tolist() for i in range(num_start_opt1)]
        start_points = [[12], [1], [13], [11]]
        kessi = 0

        next_point, next_point_aux = EI_task1.find_best_NextPoint_ei(start_points, learn_rate=0.5, num_step=800, kessi=kessi)
        # EI.plot_ei(kessis=[0.0], num_points=300, highlight_point=[next_point, next_point_aux])
        print("GP next point in task1: ", next_point)
    elif task == 2:
        # Part 2: EI on Experiment 2
        # 2.1 ZeroGP EI initialization
        EI_task2 = ExpectedImprovement()
        EI_task2.get_data_from_file("data/experiment_points_task2_gp.tsv")
        print(EI_task2.X)
        print(EI_task2.Y)

        # 2.2 AC optimization 
        # dim = EI.dim
        # low_opt1 = 0, high_opt1 = 1000
        # num_start_opt1 = 15
        # start_points = [np.random.uniform(low_opt1, high_opt1, size=dim).tolist() for i in range(num_start_opt1)]
        start_points = [[12], [1], [13], [11]]
        kessi = 0

        next_point, next_point_aux = EI_task2.find_best_NextPoint_ei(start_points, learn_rate=0.5, num_step=800, kessi=kessi)
        # EI.plot_ei(kessis=[0.0], num_points=300, highlight_point=[next_point, next_point_aux])
        print("GP next point in task2: ", next_point)        

        # Part 3: STBO on Experiments 2
        # 3.1 STBO EI initialization
        STBO = ShapeTransferBO()
        STBO.get_data_from_file("./data/experiment_points_task2_STBO.tsv")
        STBO.build_task1_gp("./data/experiment_points_task1_gp.tsv")
        STBO.build_diff_gp()

        # 3.2 AC optimization (shared the same start points as in gp)
        kessi = 0

        next_point, next_point_aux = STBO.find_best_NextPoint_ei(start_points, learn_rate=0.5, num_step=800, kessi=kessi)    
        # STBO.plot_ei(kessis=[0.0], num_points=300, highlight_point=[next_point, next_point_aux])
        print("STBO next point in task2: ", next_point)

        # Part 4: BCBO on Experiment 2
        # 4.1 BCBO EI initialization
        BCBO = BiasCorrectedBO()
        BCBO.get_data_from_file("data/experiment_points_task2_BCBO.tsv")

        print("before bias correction")
        print(BCBO.X)
        print(BCBO.Y)

        BCBO.build_task1_gp("./data/experiment_points_task1_gp.tsv")
        BCBO.build_diff_gp()

        print("after bias correction & merge")
        print(BCBO.X)
        print(BCBO.Y)

        # 4.2 AC optimization (shared the same start points as in gp)
        kessi = 0

        next_point, next_point_aux = BCBO.find_best_NextPoint_ei(start_points, learn_rate=0.5, num_step=500, kessi=kessi)    
        # BCBO.plot_ei(kessis=[0.0], num_points=300, highlight_point=[next_point, next_point_aux])
        print("BCBO next point in task2: ", next_point)
    else:
        raise(ValueError)

