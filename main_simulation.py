#!/usr/bin/env python3 

import argparse, os, random
import numpy as np

from optimization import UpperConfidenceBound
from optimization import ExpectedImprovement
from optimization import BiasCorrectedBO
from optimization import ShapeTransferBO

from simfun import exp_mu, branin, mod_branin


def arg_parser():
    "parse the argument"
    argparser = argparse.ArgumentParser(description="run simulation to compare 3 methods, ZeroGProcess, BCBO and STBO")
    argparser.add_argument("--theta", default="1.0", help="parameter in exponential target function")
    argparser.add_argument("--mu1", default="[0.0, 0.0]", help="mu of target function 1")
    argparser.add_argument("--mu2", default="[0.5, 0.5]", help="mu of target function 2")
    argparser.add_argument("--T1", default=10, help="number of experiments in target function 1")
    argparser.add_argument("--T2", default=4, help="number of experiemnts in target function 2")
    argparser.add_argument("--type", default="EXP", choices=["EXP", "BR"], help="choose target function type")

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
                results.append([(line_split[response_col], point)])

    best = max(results)[0]
    best_response = float(best[0])
    best_point = [float(ele) for ele in best[1]]

    return best_point

def main_exp(num_exp1, num_exp2, num_start_opt1=5, lr1=0.5, num_steps_opt1=500, kessi_1=0.0, file_1_gp="f1_gp.tsv",
             rand_file_1="rf1.tsv", num_start_opt2=5, lr2=0.5, num_steps_opt2=500, kessi_2=0.0, 
             file_2_gp="f2_gp.tsv", file_2_stbo="f2_stbo.tsv", file_2_bcbo="f2_bcbo.tsv", start_from_exp1=True):
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

    mu1 = [float(ele) for ele in mu1.split("_")]
    mu2 = [float(ele) for ele in mu2.split("_")]
    theta = float(theta.strip())
    
    assert(len(mu1) == len(mu2))
    dim = len(mu1)

    low = -5
    high = 5

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
        init_point_1 = np.random.uniform(low, high, size = dim)
        init_res_1 = exp_mu(init_point_1, mu1, theta)
        write_exp_result(file_1_gp, init_res_1, init_point_1)
        write_exp_result(rand_file_1, init_res_1, init_point_1)

        # run num_exp1 times on EXP 1 by random search (rand_file_1) & ZeroGP (file_1)
        if num_exp1 > 1:
            for round_k in range(num_exp1-1):
                # uniformly randomly pick next point
                next_point_rand = np.random.uniform(low, high, size = dim)
                next_response_rand = exp_mu(next_point_rand, mu1, theta)
                write_exp_result(rand_file_1, next_response_rand, next_point_rand)

                # ZeroGProcess model with EI 
                EI = ExpectedImprovement()
                EI.get_data_from_file(file_1_gp)

                start_points = [np.random.uniform(low, high, size=dim).tolist() for i in range(num_start_opt1)]

                next_point, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr1, num_step=num_steps_opt1, kessi=kessi_1)
                next_response = exp_mu(next_point, mu1, theta)
                write_exp_result(file_1_gp,  next_response, next_point)

    # Step 2: Optimization on Experiemnt 2 
    # get best point from exp1 file and get value of exp2 on best point
    best_point_exp1 = get_best_point(file_1_gp)
    res2_point_exp1 = exp_mu(best_point_exp1, mu2, theta)

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
            # Method 1: ZeroGProcess model based on EI
            EI = ExpectedImprovement()
            EI.get_data_from_file(file_2_gp)

            start_points = [np.random.uniform(low, high, size=dim).tolist() for i in range(num_start_opt2)]
            next_point, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr2, 
                                                                   num_step=num_steps_opt2, kessi=kessi_2)
            next_response = exp_mu(next_point, mu2, theta)
            write_exp_result(file_2_gp, next_response, next_point)

            # Method 2: STBO mothod based on EI from our paper
            STBO = ShapeTransferBO()
            STBO.get_data_from_file(file_2_stbo)
            STBO.build_task1_gp(file_1_gp)
            STBO.build_diff_gp()

            start_points = [np.random.uniform(low, high, size=dim).tolist() for i in range(num_start_opt2)]
            next_point, next_point_aux = STBO.find_best_NextPoint_ei(start_points, learn_rate=lr2, 
                                                                     num_step=num_steps_opt2, kessi=kessi_2)                 
            next_response = exp_mu(next_point, mu2, theta)
            write_exp_result(file_2_stbo, next_response, next_point)        

            # Method 3: BCBO method based on EI from some other paper
            BCBO = BiasCorrectedBO()
            BCBO.get_data_from_file(file_2_bcbo)
            BCBO.build_task1_gp(file_1_gp)
            BCBO.build_diff_gp()

            start_points = [np.random.uniform(low, high, size=dim).tolist() for i in range(num_start_opt2)]
            next_point, next_point_aux = BCBO.find_best_NextPoint_ei(start_points, learn_rate=lr2, 
                                                                     num_step=num_steps_opt2, kessi=kessi_2)                 
            next_response = exp_mu(next_point, mu2, theta)
            write_exp_result(file_2_bcbo, next_response, next_point)        

    return 0


def main_br(num_exp1, num_exp2, num_start_opt1=5, lr1=0.5, num_steps_opt1=500, kessi_1=0.0, file_1_gp="f1_gp.tsv",
            rand_file_1="rf1.tsv", num_start_opt2=5, lr2=0.5, num_steps_opt2=500, kessi_2=0.0, 
            file_2_gp="f2_gp.tsv", file_2_stbo="f2_stbo.tsv", file_2_bcbo="f2_bcbo.tsv", start_from_exp1=True):
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

    dim = 2 
    low = -5
    high = 5

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
        init_point_1 = np.random.uniform(low, high, size = dim)
        init_res_1 = branin(init_point_1)
        write_exp_result(file_1_gp, init_res_1, init_point_1)
        write_exp_result(rand_file_1, init_res_1, init_point_1)

        # run num_exp1 times on EXP 1 by random search (rand_file_1) & ZeroGP (file_1)
        if num_exp1 > 1:
            for round_k in range(num_exp1 - 1):
                # uniform randomly pick next point
                next_point_rand = np.random.uniform(low, high, size = dim)
                next_response_rand = branin(next_point_rand)
                write_exp_result(rand_file_1, next_response_rand, next_point_rand)

                # ZeroGProcess model with EI
                EI = ExpectedImprovement()
                EI.get_data_from_file(file_1_gp)

                start_points = [np.random.uniform(low, high, size=dim).tolist() for i in range(num_start_opt1)]

                next_point, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr1, num_step=num_steps_opt1, kessi=kessi_1)
                next_response = branin(next_point)
                write_exp_result(file_1_gp,  next_response, next_point)

    # Step 2: Optimization on Experiemnt 2 (mod_branin)
    # get best point from exp1 file and get value of exp2 on best point
    best_point_exp1 = get_best_point(file_1_gp)
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
            # Method 1: ZeroGProcess model based on EI
            EI = ExpectedImprovement()
            EI.get_data_from_file(file_2_gp)

            start_points = [np.random.uniform(low, high, size=dim).tolist() for i in range(num_start_opt2)]
            next_point, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr2, 
                                                                   num_step=num_steps_opt2, kessi=kessi_2)
            next_response = mod_branin(next_point)
            write_exp_result(file_2_gp, next_response, next_point)

            # Method 2: STBO mothod based on EI from our paper
            STBO = ShapeTransferBO()
            STBO.get_data_from_file(file_2_stbo)
            STBO.build_task1_gp(file_1_gp)
            STBO.build_diff_gp()

            start_points = [np.random.uniform(low, high, size=dim).tolist() for i in range(num_start_opt2)]
            next_point, next_point_aux = STBO.find_best_NextPoint_ei(start_points, learn_rate=lr2, 
                                                                     num_step=num_steps_opt2, kessi=kessi_2)                 
            next_response = mod_branin(next_point)
            write_exp_result(file_2_stbo, next_response, next_point)        

            # Method 3: BCBO method based on EI from some other paper
            BCBO = BiasCorrectedBO()
            BCBO.get_data_from_file(file_2_bcbo)
            BCBO.build_task1_gp(file_1_gp)
            BCBO.build_diff_gp()

            start_points = [np.random.uniform(low, high, size=dim).tolist() for i in range(num_start_opt2)]
            next_point, next_point_aux = BCBO.find_best_NextPoint_ei(start_points, learn_rate=lr2, 
                                                                     num_step=num_steps_opt2, kessi=kessi_2)                 
            next_response = mod_branin(next_point)
            write_exp_result(file_2_bcbo, next_response, next_point)

    return 0


if __name__ == "__main__":
    parser = arg_parser()

    fun_type = parser.type
    T1 = int(parser.T1)
    T2 = int(parser.T2)

    if fun_type == "EXP":
        f1_gp = "data/simExp_points_task1_gp.tsv"
        f1_rand = "data/simExp_points_task1_rand.tsv"
        f2_gp = "data/simExp_points_task2_gp.tsv"
        f2_stbo = "data/simExp_points_task2_stbo.tsv"
        f2_bcbo = "data/simExp_points_task2_bcbo.tsv"

        main_exp(T1, T2, file_1_gp=f1_gp, rand_file_1=f1_rand, file_2_gp=f2_gp, 
                 file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo, start_from_exp1=True)
        
    elif fun_type == "BR":
        f1_gp = "data/simBr_points_task1_gp.tsv"
        f1_rand = "data/simBr_points_task1_rand.tsv"
        f2_gp = "data/simBr_points_task2_gp.tsv"
        f2_stbo = "data/simBr_points_task2_stbo.tsv"
        f2_bcbo = "data/simBr_points_task2_bcbo.tsv"  
              
        main_br(T1, T2, file_1_gp=f1_gp, rand_file_1=f1_rand, file_2_gp=f2_gp, 
                file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo, start_from_exp1=True)

