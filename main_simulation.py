#!/usr/bin/env python3 

import argparse, os, random
import numpy as np

from gp import ZeroGProcess
from optimization import UpperConfidenceBound
from optimization import ExpectedImprovement
from optimization import BiasCorrectedBO
from optimization import ShapeTransferBO

from simfun import exp_mu
 

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

def main_exp(num_exp1, num_exp2, num_start_opt1=5, lr1=0.5, num_steps_opt1=500, kessi_1=0.0, file_1="f1.tsv",
             rand_file_1="rf1.tsv", num_start_opt2=5, lr2=0.5, num_steps_opt2=500, kessi_2=0.0, 
             file_2_gp="f2_gp.tsv", file_2_stbo="f2_stbo.tsv", start_from_exp1=True):
    "main function of exponential target function type"
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

    # Step 1: skip if start_from_exp1 is False
    if start_from_exp1:
        # write header & init_point to file: file_1 (ZeroGP) & rand_file_1 (random search)
        with open(file_1, "w", encoding="utf-8") as f1:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f1.writelines(header_line)

        with open(rand_file_1, "w", encoding="utf-8") as f1:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f1.writelines(header_line)

        # random initialization in exp 1
        init_point_1 = np.random.uniform(low, high, size = dim)
        init_res_1 = exp_mu(init_point_1, mu1, theta)
        write_exp_result(file_1, init_res_1, init_point_1)
        write_exp_result(rand_file_1, init_res_1, init_point_1)

        # run num_exp1 times on EXP 1 by random search (rand_file_1) & ZeroGP (file_1)
        if num_exp1 > 1:
            for round_k in range(num_exp1-1):
                # randomly pick next point
                next_point_rand = np.random.uniform(low, high, size = dim)
                next_response_rand = exp_mu(next_point_rand, mu1, theta)
                write_exp_result(rand_file_1, next_response_rand, next_point_rand)

                # ZeroGProcess model with EI 
                EI = ExpectedImprovement()
                EI.get_data_from_file(file_1)

                start_points = [np.random.uniform(low, high, size=dim).tolist() for i in range(num_start_opt1)]

                next_point, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr1, num_step=num_steps_opt1, kessi=kessi_1)
                next_response = exp_mu(next_point, mu1, theta)
                write_exp_result(file_1,  next_response, next_point)

    # Step 2: Optimization on Experiemnt 2 
    # get best point from exp1 file and get value of exp2 on best point
    best_point_exp1 = get_best_point(file_1)
    res2_point_exp1 = exp_mu(best_point_exp1, mu2, theta)

    # write header and init point for GP method
    with open(file_2_gp, "w", encoding="utf-8") as f2:
        header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
        f2.writelines(header_line)

    with open(file_2_stbo, "w", encoding="utf-8") as f2:
        header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
        f2.writelines(header_line)

    write_exp_result(file_2_gp, res2_point_exp1, best_point_exp1)
    write_exp_result(file_2_stbo, res2_point_exp1, best_point_exp1)

    if num_exp2 <=1:
        return 0

    for round_k in range(num_exp2-1):
        # Method 1: ZeroGProcess model based on EI
        EI = ExpectedImprovement()
        EI.get_data_from_file(file_2_gp)

        start_points = [np.random.uniform(low, high, size=dim).tolist() for i in range(num_start_opt2)]
        next_point, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr2, num_step=num_steps_opt2, kessi=kessi_2)
        next_response = exp_mu(next_point, mu2, theta)
        write_exp_result(file_2_gp, next_response, next_point)

        # Method 2: STBO mothod based on EI
        STBO = ShapeTransferBO()
        STBO.get_data_from_file(file_2_stbo)
        STBO.build_task1_gp(file_1)
        STBO.build_diff_gp()

        start_points = [np.random.uniform(low, high, size=dim).tolist() for i in range(num_start_opt2)]
        next_point, next_point_aux = STBO.find_best_NextPoint_ei(start_points, learn_rate=lr2, num_step=num_steps_opt2, kessi=kessi_2)                 
        next_response = exp_mu(next_point, mu2, theta)
        write_exp_result(file_2_stbo, next_response, next_point)        

        # Method 3: BCBO method based on EI
         

    return 0


def main_br(num_exp1, num_exp2):
    "main function of Brainn target function type"
    pass



if __name__ == "__main__":
    parser = arg_parser()

    fun_type = parser.type
    T1 = int(parser.T1)
    T2 = int(parser.T2)

    if fun_type == "EXP":
        f1 = "data/simulation_points_task1.tsv"
        rf1="data/simulation_points_task1_rand.tsv"
        f2_gp = "data/simulation_points_task2_gp.tsv"
        f2_stbo = "data/simulation_points_task2_stbo.tsv"

        main_exp(T1, T2, file_1=f1, rand_file_1=rf1, file_2_gp=f2_gp, file_2_stbo=f2_stbo, start_from_exp1=True)

    elif fun_type == "BR":
        main_br(T1, T2)

