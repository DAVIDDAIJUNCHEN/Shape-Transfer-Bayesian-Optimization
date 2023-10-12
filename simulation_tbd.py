#!/usr/bin/env  /mnt/users/daijun_chen/tools/miniconda3.10/install/envs/python3_huggingface/bin/python3

import argparse, os, sys, logging
import numpy as np

sys.path.append(os.getcwd())

from optimization import ExpectedImprovement

from simfun import exp_mu, branin, mod_branin

def arg_parser():
    "parse the arguments"
    argparser = argparse.ArgumentParser(description="run simulation to compare 3 methods, ZeroGProcess, BCBO and STBO")
    argparser.add_argument("--theta", default="1.0", help="parameter in exponential target function")
    argparser.add_argument("--mu1", default="[0.0, 0.0]", help="mu of target function 1")
    argparser.add_argument("--mu2", default="[0.5, 0.5]", help="mu of target function 2")
    argparser.add_argument("--T2", default=10, help="number of experiments in target function 2")
    argparser.add_argument("--type", default="EXP", choices=["EXP", "BR"], help="choose target function type")
    argparser.add_argument("--out_dir", default="./data", help="output dir")
    parser = argparser.parse_args()
    
    return parser


def write_exp_result(file, response, exp_point):
    "write experiemnt results to file"
    with open(file, 'a', encoding="utf-8") as fout:
        exp_line = str(response) + '\t' + '\t'.join([str(ele) for ele in exp_point]) + '\n'
        fout.writelines(exp_line)
    
    return 0


def main_exp(num_exp2, num_start_opt2=15, low_opt2=-5, high_opt2=10, lr2=0.5, num_steps_opt2=500, kessi_2=0.0, 
             file_2_gp_cold="f2_gp_cold.tsv"):

    theta = parser.theta
    mu1 = parser.mu1
    mu2 = parser.mu2

    mu1 = [float(ele) for ele in mu1.split("_")]
    mu2 = [float(ele) for ele in mu2.split("_")]
    theta = float(theta.strip())
    
    assert(len(mu1) == len(mu2))
    dim = len(mu1)
    
    cold_start_point = np.random.uniform(low_opt2, high_opt2, size=dim)
    res2_point_cold = exp_mu(cold_start_point, mu2, theta)

    # write header and init point
    with open(file_2_gp_cold, "w", encoding="utf-8") as f2:
        header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
        f2.writelines(header_line)            

    write_exp_result(file_2_gp_cold, res2_point_cold, cold_start_point)

    if num_exp2 > 1:
        for round_k in range(num_exp2-1):
            # all AC optimization start from the same random start points
            start_points = [np.random.uniform(low_opt2, high_opt2, size=dim).tolist() for i in range(num_start_opt2)]

            # Method 1: ZeroGProcess model based on EI
            # 1.1 GP starting from task1 best point

            # 1.2 GP with cold start point
            EI_cold = ExpectedImprovement()
            EI_cold.get_data_from_file(file_2_gp_cold)

            next_point_gp_cold, next_point_aux = EI_cold.find_best_NextPoint_ei(start_points, learn_rate=lr2,
                                                                                num_step=num_steps_opt2, kessi=kessi_2)
            next_response_gp_cold = exp_mu(next_point_gp_cold, mu2, theta)
            write_exp_result(file_2_gp_cold, next_response_gp_cold, next_point_gp_cold)

    return 0


if __name__ == "__main__":
    parser = arg_parser()

    fun_type = parser.type
    out_dir = parser.out_dir
    
    T2 = int(parser.T2)

    if fun_type == "EXP":
        f2_gp_cold = os.path.join(out_dir, "simExp_points_task2_gp" + "_from_cold" + ".tsv")

        low_opt1 = -5
        high_opt1 = 5
        low_opt2 = -5
        high_opt2 = 7

        main_exp(T2, low_opt2=low_opt2, high_opt2=high_opt2, file_2_gp_cold=f2_gp_cold)
