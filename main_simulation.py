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

def main_exp(num_exp1, num_exp2, kessi_1=0.0, num_start_opt1=5, lr1=0.5, num_steps_opt1=100):
    "main function of exponential target function type"
    theta = parser.theta
    mu1 = parser.mu1
    mu2 = parser.mu2

    mu1 = [float(ele) for ele in mu1.split("_")]
    mu2 = [float(ele) for ele in mu2.split("_")]
    theta = float(theta.strip())
    
    assert(len(mu1) == len(mu2))
    dim = len(mu1)

    # write header & init_point to file: simulation_points_task1.tsv
    with open("data/simulation_points_task1.tsv", "w", encoding="utf-8") as f1:
        header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
        f1.writelines(header_line)

    with open("data/simulation_points_task1_rand.tsv", "w", encoding="utf-8") as f1:
        header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
        f1.writelines(header_line)

    # random initialization in exp 1
    low = -5
    high = 5
    init_point_1 = np.random.uniform(low, high, size = dim)
    init_res_1 = exp_mu(init_point_1, mu1, theta)
    write_exp_result("data/simulation_points_task1_rand.tsv", init_res_1, init_point_1)
    write_exp_result("data/simulation_points_task1.tsv", init_res_1, init_point_1)

    # # run num_exp1 times on EXP 1 
    if num_exp1 > 1:
        for round_k in range(num_exp1-1):
            # randomly pick next point
            next_point_rand = np.random.uniform(low, high, size = dim)
            next_response_rand = exp_mu(next_point_rand, mu1, theta)
            write_exp_result("data/simulation_points_task1_rand.tsv", next_response_rand, next_point_rand)

            # ZeroGProcess model with EI 
            EI = ExpectedImprovement()
            EI.get_data_from_file("data/simulation_points_task1.tsv")

            start_points = [np.random.uniform(low, high, size=dim).tolist() for i in range(num_start_opt1)]

            next_point, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr1, num_step=num_steps_opt1)
            next_response = exp_mu(next_point, mu1, theta)

            write_exp_result("data/simulation_points_task1.tsv",  next_response, next_point)

    # run num_exp2 times on EXP 2



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
        main_exp(T1, T2)
    elif fun_type == "BR":
        main_br(T1, T2)

