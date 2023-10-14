#!/usr/bin/env  /mnt/users/daijun_chen/tools/miniconda3.10/install/envs/python3_huggingface/bin/python3

import argparse, os, sys, logging
import numpy as np

sys.path.append(os.getcwd())

from optimization import UpperConfidenceBound
from optimization import ExpectedImprovement
from optimization import BiasCorrectedBO
from optimization import ShapeTransferBO

from simfun import exp_mu, branin, mod_branin


def arg_parser():
    "parse the arguments"
    argparser = argparse.ArgumentParser(description="run simulation to compare 3 methods, ZeroGProcess, BCBO and STBO")
    argparser.add_argument("--theta", default="1.0", help="parameter in exponential target function")
    argparser.add_argument("--mu1", default="[0.0, 0.0]", help="mu of target function 1")
    argparser.add_argument("--mu2", default="[0.5, 0.5]", help="mu of target function 2")
    argparser.add_argument("--T1", default=10, help="number of experiments in target function 1")
    argparser.add_argument("--T2", default=4, help="number of experiemnts in target function 2")
    argparser.add_argument("--task2_start_from", default="gp", choices=["gp", "rand"], help="task2 from best point of GP/Rand task1")
    argparser.add_argument("--type", default="EXP", choices=["EXP", "BR"], help="choose target function type")
    argparser.add_argument("--from_task1", default=True, choices=['0', '1'], help="start simulation from task1")
    argparser.add_argument("--out_dir", default="./data", help="output dir")
    parser = argparser.parse_args()
    
    return parser

def write_exp_result(file, response, exp_point):
    "write experiemnt results to file"
    with open(file, 'a', encoding="utf-8") as fout:
        exp_line = str(response) + '\t' + '\t'.join([str(ele) for ele in exp_point]) + '\n'
        fout.writelines(exp_line)
    
    return 0

def get_best_point(file, response_col=0):
    "get the points with largest response"
    results = []
    with open(file, 'r', encoding="utf-8") as fin:
        for line in fin:
            if '#' not in line:
                line_split = line.split()
                point = line_split[:response_col] + line_split[(response_col+1):]
                results.append([(float(line_split[response_col]), point)])

    best = max(results)[0]
    best_response = float(best[0])
    best_point = [float(ele) for ele in best[1]]

    return best_point

def main_exp(num_exp1, num_exp2, task2_from_gp=True, num_start_opt1=5, low_opt1=-5, high_opt1=5, lr1=0.5, num_steps_opt1=500, kessi_1=0.0, file_1_gp="f1_gp.tsv",
             rand_file_1="rf1.tsv", num_start_opt2=15, low_opt2=-5, high_opt2=10, lr2=0.5, num_steps_opt2=500, kessi_2=0.0, 
             file_2_gp="f2_gp.tsv", file_2_gp_cold="f2_gp_cold.tsv", file_2_stbo="f2_stbo.tsv", file_2_bcbo="f2_bcbo.tsv"):
    """
    simulation main function of Brainn target function type:
    num_exp[1 | 2]: number of experiments in task [1 | 2]
    num_start_opt[1 | 2]: number of start points in optimizing AC function in task [1 | 2]
    lr[1 | 2]: learning rate used in optimizing AC function in task [1 | 2]
    num_steps_opt[1 | 2]: number of steps in optimizing AC function in task [1 | 2]
    kessi_[1 | 2]: kessi value used in AC function in task [1 | 2]
    file_[1 | 2]_gp: file of experiment points choosen by zeroGP in task [1 | 2]
    rand_file_1: file of experiemnt points choosen by random search in task 1
    file_2_stbo: file of experiment points choosen by our STBO in task 2
    file_2_bcbo: file of experiment points choosen by BCBO (bias corrected bayesian optimization) method
    start_from_exp1: True | False, consider False if skip experiment 1 
    """
    theta = parser.theta
    mu1 = parser.mu1
    mu2 = parser.mu2
    start_from_exp1 = int(parser.from_task1)

    mu1 = [float(ele) for ele in mu1.split("_")]
    mu2 = [float(ele) for ele in mu2.split("_")]
    theta = float(theta.strip())
    
    assert(len(mu1) == len(mu2))
    dim = len(mu1)

    # Step 1: experiment 1 (skip if start_from_exp1 is False)
    if start_from_exp1:
        # write header & init_point to file: file_1 (ZeroGP) & rand_file_1 (random search)
        with open(file_1_gp, "w", encoding="utf-8") as f1:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f1.writelines(header_line)

        with open(rand_file_1, "w", encoding="utf-8") as f1:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f1.writelines(header_line)

        # random initialization in exp 1
        init_point_1 = np.random.uniform(low_opt1, high_opt1, size = dim)
        init_res_1 = exp_mu(init_point_1, mu1, theta)
        write_exp_result(file_1_gp, init_res_1, init_point_1)
        write_exp_result(rand_file_1, init_res_1, init_point_1)

        # run num_exp1 times on EXP 1 by random search (rand_file_1) & ZeroGP (file_1)
        if num_exp1 > 1:
            for round_k in range(num_exp1-1):
                # uniformly randomly pick next point
                next_point_rand = np.random.uniform(low_opt1, high_opt1, size = dim)
                next_response_rand = exp_mu(next_point_rand, mu1, theta)
                write_exp_result(rand_file_1, next_response_rand, next_point_rand)

                # ZeroGProcess model with EI 
                EI = ExpectedImprovement()
                EI.get_data_from_file(file_1_gp)

                start_points = [np.random.uniform(low_opt1, high_opt1, size=dim).tolist() for i in range(num_start_opt1)]

                next_point, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr1, num_step=num_steps_opt1, kessi=kessi_1)
                next_response = exp_mu(next_point, mu1, theta)
                write_exp_result(file_1_gp,  next_response, next_point)

    # Step 2: Optimization on Experiemnt 2 
    # get best point from exp1 file and get value of exp2 on best point
    if task2_from_gp:  # start from best point in gp
        best_point_exp1 = get_best_point(file_1_gp)
    else:  # start from best point in random
        best_point_exp1 = get_best_point(rand_file_1)
    
    cold_start_point = np.random(low_opt2, high_opt2, size=dim)

    res2_point_exp1 = exp_mu(best_point_exp1, mu2, theta)
    res2_point_cold = exp_mu(cold_start_point, mu2, theta)

    # write header and init point
    with open(file_2_gp, "w", encoding="utf-8") as f2:
        header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
        f2.writelines(header_line)

    with open(file_2_stbo, "w", encoding="utf-8") as f2:
        header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
        f2.writelines(header_line)

    with open(file_2_bcbo, "w", encoding="utf-8") as f2:
        header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
        f2.writelines(header_line)

    with open(file_2_gp_cold, "w", encoding="utf-8") as f2:
        header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
        f2.writelines(header_line)            

    write_exp_result(file_2_gp, res2_point_exp1, best_point_exp1)
    write_exp_result(file_2_stbo, res2_point_exp1, best_point_exp1)
    write_exp_result(file_2_bcbo, res2_point_exp1, best_point_exp1)
    write_exp_result(file_2_gp_cold, res2_point_cold, cold_start_point)

    if num_exp2 > 1:
        for round_k in range(num_exp2-1):
            # all AC optimization start from the same random start points
            start_points = [np.random.uniform(low_opt2, high_opt2, size=dim).tolist() for i in range(num_start_opt2)]

            # Method 1: ZeroGProcess model based on EI
            # 1.1 GP starting from task1 best point
            EI = ExpectedImprovement()
            EI.get_data_from_file(file_2_gp)

            next_point_gp, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr2, 
                                                                   num_step=num_steps_opt2, kessi=kessi_2)
            next_response_gp = exp_mu(next_point_gp, mu2, theta)
            write_exp_result(file_2_gp, next_response_gp, next_point_gp)

            # 1.2 GP with cold start point
            EI_cold = ExpectedImprovement()
            EI_cold.get_data_from_file(file_2_gp_cold)

            next_point_gp_cold, next_point_aux = EI_cold.find_best_NextPoint_ei(start_points, learn_rate=lr2,
                                                                                num_step=num_steps_opt2, kessi=kessi_2)
            next_response_gp_cold = exp_mu(next_point_gp_cold, mu2, theta)
            write_exp_result(file_2_gp_cold, next_response_gp_cold, next_point_gp_cold)

            # Method 2: STBO mothod based on EI from our paper
            STBO = ShapeTransferBO()
            STBO.get_data_from_file(file_2_stbo)

            if task2_from_gp:   # task2 based on gp results of task1 
                STBO.build_task1_gp(file_1_gp)
            else:
                STBO.build_task1_gp(rand_file_1)
            
            STBO.build_diff_gp()

            next_point_stbo, next_point_aux = STBO.find_best_NextPoint_ei(start_points, learn_rate=lr2,
                                                                      num_step=num_steps_opt2, kessi=kessi_2)

            next_response_stbo = exp_mu(next_point_stbo, mu2, theta)
            write_exp_result(file_2_stbo, next_response_stbo, next_point_stbo)        

            # Method 3: BCBO method based on EI from some other paper
            BCBO = BiasCorrectedBO()
            BCBO.get_data_from_file(file_2_bcbo)

            if task2_from_gp:
                BCBO.build_task1_gp(file_1_gp)
            else:
                BCBO.build_task1_gp(rand_file_1)

            BCBO.build_diff_gp()

            next_point_bcbo, next_point_aux = BCBO.find_best_NextPoint_ei(start_points, learn_rate=lr2,
                                                                     num_step=num_steps_opt2, kessi=kessi_2)

            next_response_bcbo = exp_mu(next_point_bcbo, mu2, theta)
            write_exp_result(file_2_bcbo, next_response_bcbo, next_point_bcbo)        

    return 0

def main_br(num_exp1, num_exp2, task2_from_gp=True, num_start_opt1=5, low_opt1=-5, high_opt1=5, lr1=0.5, num_steps_opt1=500, kessi_1=0.0, file_1_gp="f1_gp.tsv",
            rand_file_1="rf1.tsv", num_start_opt2=5, low_opt2=-5, high_opt2=5, lr2=0.5, num_steps_opt2=500, kessi_2=0.0, 
            file_2_gp="f2_gp.tsv", file_2_stbo="f2_stbo.tsv", file_2_bcbo="f2_bcbo.tsv"):
    """
    simulation main function of Brainn target function type:
    num_exp[1 | 2]: number of experiments in task [1 | 2]
    num_start_opt[1 | 2]: number of start points in optimizing AC function in task [1 | 2]
    lr[1 | 2]: learning rate used in optimizing AC function in task [1 | 2]
    num_steps_opt[1 | 2]: number of steps in optimizing AC function in task [1 | 2]
    kessi_[1 | 2]: kessi value used in AC function in task [1 | 2]
    file_[1 | 2]_gp: file of experiment points choosen by zeroGP in task [1 | 2]
    rand_file_1: file of experiemnt points choosen by random search in task 1
    file_2_stbo: file of experiment points choosen by our STBO in task 2
    file_2_bcbo: file of experiment points choosen by BCBO (bias corrected bayesian optimization) method
    start_from_exp1: True | False, consider False if skip experiment 1 
    """

    start_from_exp1 = int(parser.from_task1)
    dim = 2

    # Step 1: experiment 1, branin (skip if start_from_exp1 is False)
    if start_from_exp1:
        # write header & init_point to file: file_1 (ZeroGP) & rand_file_1 (random search)
        with open(file_1_gp, "w", encoding="utf-8") as f1:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f1.writelines(header_line)

        with open(rand_file_1, "w", encoding="utf-8") as f1:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f1.writelines(header_line)

        # random initialization in exp 1
        init_point_1 = np.random.uniform(low_opt1, high_opt1, size = dim)
        init_res_1 = branin(init_point_1)
        write_exp_result(file_1_gp, init_res_1, init_point_1)
        write_exp_result(rand_file_1, init_res_1, init_point_1)

        # run num_exp1 times on EXP 1 by random search (rand_file_1) & ZeroGP (file_1)
        if num_exp1 > 1:
            for round_k in range(num_exp1 - 1):
                # uniform randomly pick next point
                next_point_rand = np.random.uniform(low_opt1, high_opt1, size = dim)
                next_response_rand = branin(next_point_rand)
                write_exp_result(rand_file_1, next_response_rand, next_point_rand)

                # ZeroGProcess model with EI
                EI = ExpectedImprovement()
                EI.get_data_from_file(file_1_gp)

                start_points = [np.random.uniform(low_opt1, high_opt1, size=dim).tolist() for i in range(num_start_opt1)]

                next_point, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr1, num_step=num_steps_opt1, kessi=kessi_1)
                next_response = branin(next_point)
                write_exp_result(file_1_gp,  next_response, next_point)

    # Step 2: Optimization on Experiemnt 2 (mod_branin)
    # get best point from exp1 file and get value of exp2 on best point
    if task2_from_gp:
        best_point_exp1 = get_best_point(file_1_gp)
    else:
        best_point_exp1 = get_best_point(rand_file_1)

    res2_point_exp1 = mod_branin(best_point_exp1)

    # write header and init point for GP method
    with open(file_2_gp, "w", encoding="utf-8") as f2:
        header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
        f2.writelines(header_line)

    with open(file_2_stbo, "w", encoding="utf-8") as f2:
        header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
        f2.writelines(header_line)

    with open(file_2_bcbo, "w", encoding="utf-8") as f2:
        header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
        f2.writelines(header_line)    

    write_exp_result(file_2_gp, res2_point_exp1, best_point_exp1)
    write_exp_result(file_2_stbo, res2_point_exp1, best_point_exp1)
    write_exp_result(file_2_bcbo, res2_point_exp1, best_point_exp1)

    if num_exp2 > 1:
        for round_k in range(num_exp2-1):
            # same start points are used in all AC optimization
            start_points = [np.random.uniform(low_opt2, high_opt2, size=dim).tolist() for i in range(num_start_opt2)]

            # Method 1: ZeroGProcess model based on EI
            EI = ExpectedImprovement()
            EI.get_data_from_file(file_2_gp)

            next_point_gp, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr2, 
                                                                   num_step=num_steps_opt2, kessi=kessi_2)
            next_response_gp = mod_branin(next_point_gp)
            write_exp_result(file_2_gp, next_response_gp, next_point_gp)

            # Method 2: STBO mothod based on EI from our paper
            STBO = ShapeTransferBO()
            STBO.get_data_from_file(file_2_stbo)

            if task2_from_gp:
                STBO.build_task1_gp(file_1_gp)
            else:
                STBO.build_task1_gp(rand_file_1)

            STBO.build_diff_gp()

            next_point_stbo, next_point_aux = STBO.find_best_NextPoint_ei(start_points, learn_rate=lr2, 
                                                                     num_step=num_steps_opt2, kessi=kessi_2)                 
            next_response_stbo = mod_branin(next_point_stbo)
            write_exp_result(file_2_stbo, next_response_stbo, next_point_stbo)        

            # Method 3: BCBO method based on EI from some other paper
            BCBO = BiasCorrectedBO()
            BCBO.get_data_from_file(file_2_bcbo)

            if task2_from_gp:
                BCBO.build_task1_gp(file_1_gp)
            else:
                BCBO.build_task1_gp(rand_file_1)

            BCBO.build_diff_gp()

            next_point_bcbo, next_point_aux = BCBO.find_best_NextPoint_ei(start_points, learn_rate=lr2, 
                                                                     num_step=num_steps_opt2, kessi=kessi_2)                 
            next_response_bcbo = mod_branin(next_point_bcbo)
            write_exp_result(file_2_bcbo, next_response_bcbo, next_point_bcbo)

    return 0


if __name__ == "__main__":
    parser = arg_parser()

    fun_type = parser.type
    out_dir = parser.out_dir
    
    T1 = int(parser.T1)
    T2 = int(parser.T2)
    
    task2_start_from = parser.task2_start_from

    if task2_start_from == "gp":
        task2_from_gp = True
    elif task2_start_from == "rand":
        task2_from_gp = False

    if fun_type == "EXP":
        f1_gp = os.path.join(out_dir, "simExp_points_task1_gp.tsv")
        f1_rand = os.path.join(out_dir, "simExp_points_task1_rand.tsv")

        f2_gp = os.path.join(out_dir, "simExp_points_task2_gp" + "_from_" + task2_start_from + ".tsv")
        f2_gp_cold = os.path.join(out_dir, "simExp_points_task2_gp" + "_from_cold" + ".tsv")
        f2_stbo = os.path.join(out_dir, "simExp_points_task2_stbo" + "_from_" + task2_start_from + ".tsv")
        f2_bcbo = os.path.join(out_dir, "simExp_points_task2_bcbo" + "_from_" + task2_start_from + ".tsv")

        low_opt1 = -5
        high_opt1 = 5
        low_opt2 = -5
        high_opt2 = 7

        main_exp(T1, T2, task2_from_gp, low_opt1=low_opt1, high_opt1=high_opt1, file_1_gp=f1_gp, rand_file_1=f1_rand, 
                 low_opt2=low_opt2, high_opt2=high_opt2, file_2_gp=f2_gp, file_2_gp_cold=f2_gp_cold, file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo)

    elif fun_type == "BR":
        f1_gp = os.path.join(out_dir, "simBr_points_task1_gp.tsv")
        f1_rand = os.path.join(out_dir, "simBr_points_task1_rand.tsv")

        f2_gp = os.path.join(out_dir, "simBr_points_task2_gp" + "_from_" + task2_start_from + ".tsv")
        f2_stbo = os.path.join(out_dir, "simBr_points_task2_stbo" + "_from_" + task2_start_from + ".tsv")
        f2_bcbo = os.path.join(out_dir, "simBr_points_task2_bcbo" + "_from_" + task2_start_from + ".tsv")

        low_opt1 = -5
        high_opt1 = 5
        low_opt2 = -5
        high_opt2 = 5

        main_br(T1, T2, task2_from_gp, low_opt1=low_opt1, high_opt1=high_opt1, file_1_gp=f1_gp, rand_file_1=f1_rand, 
                low_opt2=low_opt2, high_opt2=high_opt2, file_2_gp=f2_gp, file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo)

