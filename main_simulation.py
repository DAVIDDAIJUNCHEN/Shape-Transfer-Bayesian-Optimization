#!/usr/bin/env  /mnt/users/daijun_chen/tools/miniconda3.10/install/envs/python3_huggingface/bin/python3

import argparse, os, sys, logging
import numpy as np

sys.path.append(os.getcwd())

from gp import ZeroGProcess
from optimization import UpperConfidenceBound
from optimization import ExpectedImprovement
from optimization import BiasCorrectedBO
from optimization import ShapeTransferBO

from simfun import exp_mu, branin, mod_branin, needle_func, mono_func, two_exp_mu, tri_exp_mu
from utils import write_exp_result, get_best_point


def arg_parser():
    "parse the arguments"
    argparser = argparse.ArgumentParser(description="run simulation to compare 3 methods, ZeroGProcess, BCBO and STBO")
    argparser.add_argument("--type", default="EXP", choices=["EXP", "BR", "NEEDLE", "MONO2NEEDLE", "MONO2DOUBLE", "DOUBLE2DOUBLE", "TRIPLE2DOUBLE", "DOUBLE2TRIPLE"], help="choose target function type")

    # arguments for EXP type only
    argparser.add_argument("--theta", default="1.0", help="shape parameter in tyep EXP")
    argparser.add_argument("--mu1", default="0.0_0.0", help="scale parameter of target function 1 in type EXP")
    argparser.add_argument("--mu2", default="0.5_0.5", help="scale parameter of target function 2 in type EXP")

    # arguments for NEEDLE type only
    argparser.add_argument("--needle_shift", default="0.3", help="shift parameter of target function 2 in type NEEDLE")

    # general arguments 
    argparser.add_argument("--T1", default="10", help="number of experiments in target function 1")
    argparser.add_argument("--T2", default="4",  help="number of experiemnts in target function 2")
    argparser.add_argument("--task2_start_from", default="gp", choices=["gp", "rand"], help="task2 from best point of GP/Rand task1")
    argparser.add_argument("--from_task1", default=True, choices=['0', '1', '2'], help="start simulation from task1 (use existing task1 results, or run task1 only)")
    argparser.add_argument("--out_dir", default="./data", help="output dir")

    parser = argparser.parse_args()
    
    return parser

def main_experiment(num_exp1, num_exp2, task2_from_gp=True, num_start_opt1=50, low_opt1=-5, high_opt1=5, lr1=0.5, num_steps_opt1=50, kessi_1=0.0, 
             file_1_gp="f1_gp.tsv", file_1_rand="f1_rand.tsv", file_1_sample="f1_sample.tsv", file_1_mean="f1_mean.tsv", 
             file_1_sample_stbo="f1_sample_stbo.tsv", file_1_mean_stbo="f1_mean_stbo.tsv",  
             num_start_opt2=25, low_opt2=-5, high_opt2=10, lr2=0.5, num_steps_opt2=200, kessi_2=0.0, 
             file_2_gp="f2_gp.tsv", file_2_gp_cold="f2_gp_cold.tsv", file_2_stbo="f2_stbo.tsv", file_2_bcbo="f2_bcbo.tsv", fun_type="EXP"):
    """
    simulation main function:
    num_exp[1 | 2]: number of experiments in task [1 | 2]
    num_start_opt[1 | 2]: number of start points in optimizing AC function in task [1 | 2]
    lr[1 | 2]: learning rate used in optimizing AC function in task [1 | 2]
    num_steps_opt[1 | 2]: number of steps in optimizing AC function in task [1 | 2]
    kessi_[1 | 2]: kessi value used in AC function in task [1 | 2]
    file_[1 | 2]_gp: file of experiment points choosen by zeroGP in task [1 | 2]
    file_2_gp_cold: file of experiment points choosen by zeroGP from cold start in task 2
    file_1_rand: file of experiemnt points choosen by random search in task 1
    file_1_sample: file of experiment points sampled from Gaussian process
    file_1_sample_stbo: file of experiment points choosen by STBO (on file_1_sample) in task 1
    file_2_stbo: file of experiment points choosen by our STBO in task 2
    file_2_bcbo: file of experiment points choosen by BCBO (bias corrected bayesian optimization) method
    start_from_exp1: True | False, consider False if skip experiment 1 
    """
    start_from_exp1 = int(parser.from_task1)

    if fun_type == "EXP":
        theta = parser.theta
        mu1 = parser.mu1
        mu2 = parser.mu2

        theta = float(theta.strip())
        mu1 = [float(ele) for ele in mu1.split("_")]
        mu2 = [float(ele) for ele in mu2.split("_")]
        
        assert(len(mu1) == len(mu2))
        dim = len(mu1)
    elif fun_type == "BR":
        dim = 2
    elif fun_type == "NEEDLE":
        needle_shift = float(parser.needle_shift)
        dim = 1
    elif fun_type == "MONO2NEEDLE":
        needle_shift = float(parser.needle_shift)
        dim = 1
    elif fun_type == "MONO2DOUBLE":
        dim = 1
    elif fun_type == "DOUBLE2DOUBLE":
        dim = 1
        lambda1 = 1; lambda2 = 1.5
        mu1 = [0]; mu2 = [5]
        theta1 = 1; theta2 = 1
    elif fun_type == "TRIPLE2DOUBLE":
        dim = 1
        lambda1 = 1; lambda2 = 1.5; lambda3 = 1.25
        mu1 = [0]; mu2 = [5]; mu3 = [10]
        lambda1_t2 = lambda2 + 0.2; lambda2_t2 = 0; lambda3_t2 = lambda1 - 0.2
        mu1_t2 = [mu1[0]+0.2]; mu2_t2 = mu2; mu3_t2 = [mu3[0]-0.2]
        theta1 = 1; theta2 = 1; theta3 = 1
    elif fun_type == "DOUBLE2TRIPLE":
        dim = 1

        lambda1 = 1.7; lambda2 = 0; lambda3 = 0.8
        mu1 = [0.1]; mu2 = [5]; mu3 = [9.9]
        
        lambda1_t2 = 1; lambda2_t2 = 1.4; lambda3_t2 = 1.9
        mu1_t2 = [0]; mu2_t2 = [5]; mu3_t2 = [10]

        theta1 = 1; theta2 = 1; theta3 = 1
    else:
        raise(TypeError)

    # Step 1: experiment 1 (skip if start_from_exp1 is 0, run if start_from_exp1 is 1 or 2)
    if start_from_exp1:
        # write header & init_point to file: file_1 (ZeroGP) & rand_file_1 (random search) & file_1_sample_stbo
        with open(file_1_gp, "w", encoding="utf-8") as f1:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f1.writelines(header_line)

        with open(file_1_rand, "w", encoding="utf-8") as f1:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f1.writelines(header_line)

        with open(file_1_sample_stbo, "w", encoding="utf-8") as f1:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f1.writelines(header_line)
        
        with open(file_1_mean_stbo, "w", encoding="utf-8") as f1:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f1.writelines(header_line)

        # Method 3 in task 1: GP-based Sampling STBO
        # stage 1: sampling from Gaussian Process
        zeroGP = ZeroGProcess()
        zeroGP.get_data_from_file(file_1_sample_stbo)

        num_sample = 10
        mean_sample = 0.75
        sigma_sample = 0.01

        lower_bound = [low_opt1 for i in range(dim)]
        upper_bound = [high_opt1 for i in range(dim)]

        # give different weak prior information 
        if fun_type == "DOUBLE2DOUBLE":
            prior_pnts = [([0.15], 1.1), ([2.8], 0.5)]
        elif fun_type == "DOUBLE2TRIPLE":
            prior_pnts = [([0.2], 1.5), ([5.5], 0.5), ([11], 1.5)]
        elif fun_type == "TRIPLE2DOUBLE":
            prior_pnts = [([-0.2], 1.1), ([5.5], 1.1), ([7.5], 0.5), ([11], 1.1)]

        zeroGP.sample(num_sample, mean_sample, sigma_sample, l_bounds=lower_bound, u_bounds=upper_bound, prior_points=prior_pnts, mean_fix=False, out_file=file_1_sample)
        best_point_exp0_sample = get_best_point(file_1_sample)

        # Method 4 in task 1: mean reduction STBO
        zeroGP.sample(num_sample, mean_sample, sigma_sample, l_bounds=lower_bound, u_bounds=upper_bound, prior_points=prior_pnts, mean_fix=True, out_file=file_1_mean)
        best_point_exp0_mean = get_best_point(file_1_mean)

        # Task 1: random initialization & best point initialization from GP sample
        init_point_1 = np.random.uniform(low_opt1, high_opt1, size=dim)

        if fun_type == "EXP":
            init_res_1 = exp_mu(init_point_1, mu1, theta)
            res1_point_exp0_sample = exp_mu(best_point_exp0_sample, mu1, theta)
            res1_point_exp0_mean = exp_mu(best_point_exp0_mean, mu1, theta)
        elif fun_type == "BR":
            init_res_1 = branin(init_point_1)
            res1_point_exp0_sample = branin(best_point_exp0_sample)
            res1_point_exp0_mean = branin(best_point_exp0_mean)
        elif fun_type == "NEEDLE":
            init_res_1 = needle_func(init_point_1, shift=0)
            res1_point_exp0_sample = needle_func(best_point_exp0_sample, shift=0)
            res1_point_exp0_mean = needle_func(best_point_exp0_mean, shift=0)
        elif fun_type == "MONO2NEEDLE":
            init_res_1 = mono_func(init_point_1)
            res1_point_exp0_sample = mono_func(best_point_exp0_sample)
            res1_point_exp0_mean = mono_func(best_point_exp0_mean)
        elif fun_type == "MONO2DOUBLE":
            init_res_1 = exp_mu(init_point_1, [0], 0.5)
            res1_point_exp0_sample = exp_mu(best_point_exp0_sample, [0], 0.5)  
            res1_point_exp0_mean = exp_mu(best_point_exp0_mean, [0], 0.5)  
        elif fun_type == "DOUBLE2DOUBLE":
            init_res_1 = two_exp_mu(init_point_1, lambda1, lambda2, mu1, mu2, theta1, theta2)
            res1_point_exp0_sample = two_exp_mu(best_point_exp0_sample, lambda1, lambda2, mu1, mu2, theta1, theta2)
            res1_point_exp0_mean = two_exp_mu(best_point_exp0_mean, lambda1, lambda2, mu1, mu2, theta1, theta2)
        elif fun_type == "TRIPLE2DOUBLE":
            init_res_1 = tri_exp_mu(init_point_1, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
            res1_point_exp0_sample = tri_exp_mu(best_point_exp0_sample, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)   
            res1_point_exp0_mean = tri_exp_mu(best_point_exp0_mean, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
        elif fun_type == "DOUBLE2TRIPLE":
            init_res_1 = tri_exp_mu(init_point_1, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
            res1_point_exp0_sample = tri_exp_mu(best_point_exp0_sample, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
            res1_point_exp0_mean = tri_exp_mu(best_point_exp0_mean, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
        else:
            raise(TypeError)

        write_exp_result(file_1_gp, init_res_1, init_point_1)
        write_exp_result(file_1_rand, init_res_1, init_point_1)
        write_exp_result(file_1_sample_stbo, res1_point_exp0_sample, best_point_exp0_sample)
        write_exp_result(file_1_mean_stbo, res1_point_exp0_mean, best_point_exp0_mean)

        # run num_exp1 times on EXP 1 by random search (rand_file_1) & ZeroGP (file_1)
        if num_exp1 > 1:
            for round_k in range(num_exp1-1):
                # Method 1: uniformly randomly pick next point
                next_point_rand = np.random.uniform(low_opt1, high_opt1, size=dim)

                start_points = [np.random.uniform(low_opt1, high_opt1, size=dim).tolist() for i in range(num_start_opt1)]
                # Method 2: ZeroGProcess model with EI
                EI = ExpectedImprovement()
                EI.get_data_from_file(file_1_gp)

                next_point_ei, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr1, num_step=num_steps_opt1, kessi=kessi_1)
                                
                # Method 3: GP-based Sampling STBO
                STBO_task1_sample = ShapeTransferBO()
                STBO_task1_sample.get_data_from_file(file_1_sample_stbo)
                STBO_task1_sample.build_task1_gp(file_1_sample)
                STBO_task1_sample.build_diff_gp()

                next_point_stbo1_sample, next_point_aux = STBO_task1_sample.find_best_NextPoint_ei(start_points, learn_rate=lr1, 
                                                                            num_step=num_steps_opt1, kessi=kessi_1)

                # Method 4: mean reduction STBO
                STBO_task1_mean = ShapeTransferBO()
                STBO_task1_mean.get_data_from_file(file_1_mean_stbo)
                STBO_task1_mean.build_task1_gp(file_1_mean)
                STBO_task1_mean.build_diff_gp()

                next_point_stbo1_mean, next_point_aux = STBO_task1_mean.find_best_NextPoint_ei(start_points, learn_rate=lr1, 
                                                                            num_step=num_steps_opt1, kessi=kessi_1)                

                if fun_type == "EXP":
                    next_response_rand  = exp_mu(next_point_rand, mu1, theta)
                    next_response_ei    = exp_mu(next_point_ei, mu1, theta)
                    next_response_stbo1_sample = exp_mu(next_point_stbo1_sample, mu1, theta)
                    next_response_stbo1_mean = exp_mu(next_point_stbo1_mean, mu1, theta)
                elif fun_type == "BR":
                    next_response_rand  = branin(next_point_rand)
                    next_response_ei    = branin(next_point_ei)
                    next_response_stbo1_sample = branin(next_point_stbo1_sample)
                    next_response_stbo1_mean = branin(next_point_stbo1_mean)
                elif fun_type == "NEEDLE":
                    next_response_rand   = needle_func(next_point_rand, shift=0)
                    next_response_ei     = needle_func(next_point_ei, shift=0)
                    next_response_stbo1_sample  = needle_func(next_point_stbo1_sample, shift=0)
                    next_response_stbo1_mean  = needle_func(next_point_stbo1_mean, shift=0)
                elif fun_type == "MONO2NEEDLE":
                    next_response_rand   = mono_func(next_point_rand)
                    next_response_ei     = mono_func(next_point_ei)
                    next_response_stbo1_sample  = mono_func(next_point_stbo1_sample)
                    next_response_stbo1_mean  = mono_func(next_point_stbo1_mean)
                elif fun_type == "MONO2DOUBLE":
                    next_response_rand   = exp_mu(next_point_rand, [0], 0.5)
                    next_response_ei     = exp_mu(next_point_ei, [0], 0.5)
                    next_response_stbo1_sample  = exp_mu(next_point_stbo1_sample, [0], 0.5)
                    next_response_stbo1_mean  = exp_mu(next_point_stbo1_mean, [0], 0.5)
                elif fun_type == "DOUBLE2DOUBLE":
                    next_response_rand   = two_exp_mu(next_point_rand, lambda1, lambda2, mu1, mu2, theta1, theta2)
                    next_response_ei     = two_exp_mu(next_point_ei, lambda1, lambda2, mu1, mu2, theta1, theta2)
                    next_response_stbo1_sample  = two_exp_mu(next_point_stbo1_sample, lambda1, lambda2, mu1, mu2, theta1, theta2)
                    next_response_stbo1_mean  = two_exp_mu(next_point_stbo1_mean, lambda1, lambda2, mu1, mu2, theta1, theta2)
                elif fun_type == "TRIPLE2DOUBLE":
                    next_response_rand   = tri_exp_mu(next_point_rand, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                    next_response_ei     = tri_exp_mu(next_point_ei, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                    next_response_stbo1_sample  = tri_exp_mu(next_point_stbo1_sample, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                    next_response_stbo1_mean  = tri_exp_mu(next_point_stbo1_mean, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                elif fun_type == "DOUBLE2TRIPLE":
                    next_response_rand   = tri_exp_mu(next_point_rand, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                    next_response_ei     = tri_exp_mu(next_point_ei, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                    next_response_stbo1_sample  = tri_exp_mu(next_point_stbo1_sample, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                    next_response_stbo1_mean  = tri_exp_mu(next_point_stbo1_mean, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                else:
                    raise(TypeError)
                
                write_exp_result(file_1_rand, next_response_rand, next_point_rand)
                write_exp_result(file_1_gp,  next_response_ei, next_point_ei)
                write_exp_result(file_1_sample_stbo,  next_response_stbo1_sample, next_point_stbo1_sample)
                write_exp_result(file_1_mean_stbo,  next_response_stbo1_mean, next_point_stbo1_mean)
    
    # Skip experiment 2 if start_from_exp1 = 2
    if start_from_exp1 == 2:
        return 0

    # Step 2: Optimization on Experiemnt 2 
    # get best point from exp1 file and get value of exp2 on best point
    if task2_from_gp:  # start from best point in gp
        best_point_exp1 = get_best_point(file_1_gp)
    else:              # start from best point in random
        best_point_exp1 = get_best_point(file_1_rand)
    
    cold_start_point = np.random.uniform(low_opt2, high_opt2, size=dim)

    if fun_type == "EXP":
        res2_point_exp1 = exp_mu(best_point_exp1, mu2, theta)
        res2_point_cold = exp_mu(cold_start_point, mu2, theta)
    elif fun_type == "BR":
        res2_point_exp1 = mod_branin(best_point_exp1)
        res2_point_cold = mod_branin(cold_start_point)
    elif fun_type == "NEEDLE":
        res2_point_exp1 = needle_func(best_point_exp1, shift=needle_shift)
        res2_point_cold = needle_func(cold_start_point, shift=needle_shift)
    elif fun_type == "MONO2NEEDLE":
        res2_point_exp1 = needle_func(best_point_exp1, shift=needle_shift)
        res2_point_cold = needle_func(cold_start_point, shift=needle_shift)    
    elif fun_type == "MONO2DOUBLE":
        res2_point_exp1 = two_exp_mu(best_point_exp1, mu1=[0], mu2=[9], theta1=0.5, theta2=2)
        res2_point_cold = two_exp_mu(cold_start_point, mu1=[0], mu2=[9], theta1=0.5, theta2=2)    
    elif fun_type == "DOUBLE2DOUBLE":
        res2_point_exp1 = two_exp_mu(best_point_exp1, lambda2, lambda1, mu1, mu2, theta1, theta2)
        res2_point_cold = two_exp_mu(cold_start_point, lambda2, lambda1, mu1, mu2, theta1, theta2)    
    elif fun_type == "TRIPLE2DOUBLE":
        res2_point_exp1 = tri_exp_mu(best_point_exp1, lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3)
        res2_point_cold = tri_exp_mu(cold_start_point, lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3)   
    elif fun_type == "DOUBLE2TRIPLE":
        res2_point_exp1 = tri_exp_mu(best_point_exp1, lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3)
        res2_point_cold = tri_exp_mu(cold_start_point, lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3)      
    else:
        raise(TypeError)

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

    if not task2_from_gp:   # run task2 from cold when other methods start from rand
        with open(file_2_gp_cold, "w", encoding="utf-8") as f2:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f2.writelines(header_line)

    write_exp_result(file_2_gp, res2_point_exp1, best_point_exp1)
    write_exp_result(file_2_stbo, res2_point_exp1, best_point_exp1)
    write_exp_result(file_2_bcbo, res2_point_exp1, best_point_exp1)
    
    if not task2_from_gp:   # run task2 from cold when other methods start from rand
        write_exp_result(file_2_gp_cold, res2_point_cold, cold_start_point)  # start point from cold not exp1

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
            if fun_type == "EXP":
                next_response_gp = exp_mu(next_point_gp, mu2, theta)
            elif fun_type == "BR":
                next_response_gp = mod_branin(next_point_gp)
            elif fun_type == "NEEDLE":
                next_response_gp = needle_func(next_point_gp, shift=needle_shift)
            elif fun_type == "MONO2NEEDLE":
                next_response_gp = needle_func(next_point_gp, shift=needle_shift)
            elif fun_type == "MONO2DOUBLE":
                next_response_gp = two_exp_mu(next_point_gp, mu1=[0], mu2=[9], theta1=0.5, theta2=2)
            elif fun_type == "DOUBLE2DOUBLE":
                next_response_gp = two_exp_mu(next_point_gp, lambda2, lambda1, mu1, mu2, theta1, theta2)
            elif fun_type == "TRIPLE2DOUBLE":
                next_response_gp = tri_exp_mu(next_point_gp, lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3)
            elif fun_type == "DOUBLE2TRIPLE":
                next_response_gp = tri_exp_mu(next_point_gp, lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3)
            else:
                raise(TypeError)

            write_exp_result(file_2_gp, next_response_gp, next_point_gp)

            # 1.2 GP with cold start point
            if not task2_from_gp:   # when other methods start from rand
                EI_cold = ExpectedImprovement()
                EI_cold.get_data_from_file(file_2_gp_cold)

                next_point_gp_cold, next_point_aux = EI_cold.find_best_NextPoint_ei(start_points, learn_rate=lr2,
                                                                                num_step=num_steps_opt2, kessi=kessi_2)
                if fun_type == "EXP":
                    next_response_gp_cold = exp_mu(next_point_gp_cold, mu2, theta)
                elif fun_type == "BR":
                    next_response_gp_cold = mod_branin(next_point_gp_cold)
                elif fun_type == "NEEDLE":
                    next_response_gp_cold = needle_func(next_point_gp_cold, shift=needle_shift)
                elif fun_type == "MONO2NEEDLE":
                    next_response_gp_cold = needle_func(next_point_gp_cold, shift=needle_shift)
                elif fun_type == "MONO2DOUBLE":
                    next_response_gp_cold = two_exp_mu(next_point_gp_cold, mu1=[0], mu2=[9], theta1=0.5, theta2=2)
                elif fun_type == "DOUBLE2DOUBLE":
                    next_response_gp_cold = two_exp_mu(next_point_gp_cold, lambda2, lambda1, mu1, mu2, theta1, theta2)
                elif fun_type == "TRIPLE2DOUBLE":
                    next_response_gp_cold = tri_exp_mu(next_point_gp_cold, lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3)
                elif fun_type == "DOUBLE2TRIPLE":
                    next_response_gp_cold = tri_exp_mu(next_point_gp_cold, lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3)
                else:
                    raise(TypeError)

                write_exp_result(file_2_gp_cold, next_response_gp_cold, next_point_gp_cold)

            # Method 2: STBO mothod based on EI from our paper
            STBO = ShapeTransferBO()
            STBO.get_data_from_file(file_2_stbo)

            if task2_from_gp:   # task2 based on gp results of task1 
                STBO.build_task1_gp(file_1_gp)
            else:
                STBO.build_task1_gp(file_1_rand)
            
            STBO.build_diff_gp()

            next_point_stbo, next_point_aux = STBO.find_best_NextPoint_ei(start_points, learn_rate=lr2,
                                                                      num_step=num_steps_opt2, kessi=kessi_2)

            if fun_type == "EXP":
                next_response_stbo = exp_mu(next_point_stbo, mu2, theta)
            elif fun_type == "BR":
                next_response_stbo = mod_branin(next_point_stbo)
            elif fun_type == "NEEDLE":
                next_response_stbo = needle_func(next_point_stbo, shift=needle_shift)
            elif fun_type == "MONO2NEEDLE":
                next_response_stbo = needle_func(next_point_stbo, shift=needle_shift)
            elif fun_type == "MONO2DOUBLE":
                next_response_stbo = two_exp_mu(next_point_stbo, mu1=[0], mu2=[9], theta1=0.5, theta2=2)
            elif fun_type == "DOUBLE2DOUBLE":
                next_response_stbo = two_exp_mu(next_point_stbo, lambda2, lambda1, mu1, mu2, theta1, theta2)
            elif fun_type == "TRIPLE2DOUBLE":
                next_response_stbo = tri_exp_mu(next_point_stbo,  lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3)
            elif fun_type == "DOUBLE2TRIPLE":
                next_response_stbo = tri_exp_mu(next_point_stbo,  lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3)
            else:
                raise(TypeError)

            write_exp_result(file_2_stbo, next_response_stbo, next_point_stbo)

            # Method 3: BCBO method based on EI from some other paper
            BCBO = BiasCorrectedBO()
            BCBO.get_data_from_file(file_2_bcbo)

            if task2_from_gp:
                BCBO.build_task1_gp(file_1_gp)
            else:
                BCBO.build_task1_gp(file_1_rand)

            BCBO.build_diff_gp()

            next_point_bcbo, next_point_aux = BCBO.find_best_NextPoint_ei(start_points, learn_rate=lr2,
                                                                     num_step=num_steps_opt2, kessi=kessi_2)

            if fun_type == "EXP":
                next_response_bcbo = exp_mu(next_point_bcbo, mu2, theta)
            elif fun_type == "BR":
                next_response_bcbo = mod_branin(next_point_bcbo)
            elif fun_type == "NEEDLE":
                next_response_bcbo = needle_func(next_point_bcbo, shift=needle_shift)
            elif fun_type == "MONO2NEEDLE":
                next_response_bcbo = needle_func(next_point_bcbo, shift=needle_shift)
            elif fun_type == "MONO2DOUBLE":
                next_response_bcbo = two_exp_mu(next_point_bcbo, mu1=[0], mu2=[9], theta1=0.5, theta2=2)
            elif fun_type == "DOUBLE2DOUBLE":
                next_response_bcbo = two_exp_mu(next_point_bcbo, lambda2, lambda1, mu1, mu2, theta1, theta2)
            elif fun_type == "TRIPLE2DOUBLE":
                next_response_bcbo = tri_exp_mu(next_point_bcbo, lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3)
            elif fun_type == "DOUBLE2TRIPLE":
                next_response_bcbo = tri_exp_mu(next_point_bcbo, lambda1_t2, lambda2_t2, lambda3_t2, mu1_t2, mu2_t2, mu3_t2, theta1, theta2, theta3)
            else:
                raise(TypeError)

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
        f1_sample = os.path.join(out_dir, "simExp_points_task0_sample.tsv")
        f1_mean = os.path.join(out_dir, "simExp_points_task0_mean.tsv")
        f1_sample_stbo = os.path.join(out_dir, "simExp_points_task1_sample_stbo.tsv")
        f1_mean_stbo = os.path.join(out_dir, "simExp_points_task1_mean_stbo.tsv")

        f2_gp = os.path.join(out_dir, "simExp_points_task2_gp" + "_from_" + task2_start_from + ".tsv")
        f2_gp_cold = os.path.join(out_dir, "simExp_points_task2_gp" + "_from_cold" + ".tsv")
        f2_stbo = os.path.join(out_dir, "simExp_points_task2_stbo" + "_from_" + task2_start_from + ".tsv")
        f2_bcbo = os.path.join(out_dir, "simExp_points_task2_bcbo" + "_from_" + task2_start_from + ".tsv")

        low_opt1 = -5
        high_opt1 = 5
        low_opt2 = -5
        high_opt2 = 7

        main_experiment(T1, T2, task2_from_gp, low_opt1=low_opt1, high_opt1=high_opt1, file_1_gp=f1_gp, file_1_rand=f1_rand, 
                file_1_sample=f1_sample, file_1_mean=f1_mean, file_1_sample_stbo=f1_sample_stbo, file_1_mean_stbo=f1_mean_stbo, 
                fun_type="EXP", low_opt2=low_opt2, high_opt2=high_opt2, file_2_gp=f2_gp, file_2_gp_cold=f2_gp_cold, 
                file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo)

    elif fun_type == "BR":
        f1_gp = os.path.join(out_dir, "simBr_points_task1_gp.tsv")
        f1_rand = os.path.join(out_dir, "simBr_points_task1_rand.tsv")
        f1_sample = os.path.join(out_dir, "simBr_points_task0_sample.tsv")
        f1_mean = os.path.join(out_dir, "simBr_points_task0_mean.tsv")
        f1_sample_stbo = os.path.join(out_dir, "simBr_points_task1_sample_stbo.tsv")
        f1_mean_stbo = os.path.join(out_dir, "simBr_points_task1_mean_stbo.tsv")

        f2_gp = os.path.join(out_dir, "simBr_points_task2_gp" + "_from_" + task2_start_from + ".tsv")
        f2_gp_cold = os.path.join(out_dir, "simBr_points_task2_gp" + "_from_cold" + ".tsv")
        f2_stbo = os.path.join(out_dir, "simBr_points_task2_stbo" + "_from_" + task2_start_from + ".tsv")
        f2_bcbo = os.path.join(out_dir, "simBr_points_task2_bcbo" + "_from_" + task2_start_from + ".tsv")

        low_opt1 = -10
        high_opt1 = 10
        low_opt2 = -10
        high_opt2 = 10

        main_experiment(T1, T2, task2_from_gp, low_opt1=low_opt1, high_opt1=high_opt1, file_1_gp=f1_gp, file_1_rand=f1_rand, 
                file_1_sample=f1_sample, file_1_mean=f1_mean, file_1_sample_stbo=f1_sample_stbo, file_1_mean_stbo=f1_mean_stbo,
                fun_type="BR", low_opt2=low_opt2, high_opt2=high_opt2, file_2_gp=f2_gp, file_2_gp_cold=f2_gp_cold, 
                file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo)

    elif fun_type == "NEEDLE":
        f1_gp = os.path.join(out_dir, "simNeedle_points_task1_gp.tsv")
        f1_rand = os.path.join(out_dir, "simNeedle_points_task1_rand.tsv")
        f1_sample = os.path.join(out_dir, "simNeedle_points_task0_sample.tsv")
        f1_mean = os.path.join(out_dir, "simNeedle_points_task0_mean.tsv")
        f1_sample_stbo = os.path.join(out_dir, "simNeedle_points_task1_sample_stbo.tsv")
        f1_mean_stbo = os.path.join(out_dir, "simNeedle_points_task1_mean_stbo.tsv")

        f2_gp = os.path.join(out_dir, "simNeedle_points_task2_gp" + "_from_" + task2_start_from + ".tsv")
        f2_gp_cold = os.path.join(out_dir, "simNeedle_points_task2_gp" + "_from_cold" + ".tsv")
        f2_stbo = os.path.join(out_dir, "simNeedle_points_task2_stbo" + "_from_" + task2_start_from + ".tsv")
        f2_bcbo = os.path.join(out_dir, "simNeedle_points_task2_bcbo" + "_from_" + task2_start_from + ".tsv")

        low_opt1 = 0
        high_opt1 = 10
        low_opt2 = 0
        high_opt2 = 10

        main_experiment(T1, T2, task2_from_gp, low_opt1=low_opt1, high_opt1=high_opt1, file_1_gp=f1_gp, file_1_rand=f1_rand, 
                file_1_sample=f1_sample, file_1_mean=f1_mean, file_1_sample_stbo=f1_sample_stbo, file_1_mean_stbo=f1_mean_stbo, 
                fun_type="NEEDLE", low_opt2=low_opt2, high_opt2=high_opt2, file_2_gp=f2_gp, file_2_gp_cold=f2_gp_cold, 
                file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo)

    elif fun_type == "MONO2NEEDLE":
        f1_gp = os.path.join(out_dir, "simMono2Needle_points_task1_gp.tsv")
        f1_rand = os.path.join(out_dir, "simMono2Needle_points_task1_rand.tsv")
        f1_sample = os.path.join(out_dir, "simMono2Needle_points_task0_sample.tsv")
        f1_mean = os.path.join(out_dir, "simMono2Needle_points_task0_mean.tsv")
        f1_sample_stbo = os.path.join(out_dir, "simMono2Needle_points_task1_sample_stbo.tsv")
        f1_mean_stbo = os.path.join(out_dir, "simMono2Needle_points_task1_mean_stbo.tsv")

        f2_gp = os.path.join(out_dir, "simMono2Needle_points_task2_gp" + "_from_" + task2_start_from + ".tsv")
        f2_gp_cold = os.path.join(out_dir, "simMono2Needle_points_task2_gp" + "_from_cold" + ".tsv")
        f2_stbo = os.path.join(out_dir, "simMono2Needle_points_task2_stbo" + "_from_" + task2_start_from + ".tsv")
        f2_bcbo = os.path.join(out_dir, "simMono2Needle_points_task2_bcbo" + "_from_" + task2_start_from + ".tsv")

        low_opt1 = 0
        high_opt1 = 10
        low_opt2 = 0
        high_opt2 = 10

        main_experiment(T1, T2, task2_from_gp, low_opt1=low_opt1, high_opt1=high_opt1, file_1_gp=f1_gp, file_1_rand=f1_rand, 
                file_1_sample=f1_sample, file_1_mean=f1_mean, file_1_sample_stbo=f1_sample_stbo, file_1_mean_stbo=f1_mean_stbo,
                fun_type="MONO2NEEDLE", low_opt2=low_opt2, high_opt2=high_opt2, file_2_gp=f2_gp, file_2_gp_cold=f2_gp_cold, 
                file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo)

    elif fun_type == "MONO2DOUBLE":
        f1_gp = os.path.join(out_dir, "simMono2Double_points_task1_gp.tsv")
        f1_rand = os.path.join(out_dir, "simMono2Double_points_task1_rand.tsv")
        f1_sample = os.path.join(out_dir, "simMono2Double_points_task0_sample.tsv")
        f1_mean = os.path.join(out_dir, "simMono2Double_points_task0_mean.tsv")
        f1_sample_stbo = os.path.join(out_dir, "simMono2Double_points_task1_sample_stbo.tsv")
        f1_mean_stbo = os.path.join(out_dir, "simMono2Double_points_task1_mean_stbo.tsv")

        f2_gp = os.path.join(out_dir, "simMono2Double_points_task2_gp" + "_from_" + task2_start_from + ".tsv")
        f2_gp_cold = os.path.join(out_dir, "simMono2Double_points_task2_gp" + "_from_cold" + ".tsv")
        f2_stbo = os.path.join(out_dir, "simMono2Double_points_task2_stbo" + "_from_" + task2_start_from + ".tsv")
        f2_bcbo = os.path.join(out_dir, "simMono2Double_points_task2_bcbo" + "_from_" + task2_start_from + ".tsv")

        low_opt1 = -5
        high_opt1 = 15
        low_opt2 = -5
        high_opt2 = 15

        main_experiment(T1, T2, task2_from_gp, low_opt1=low_opt1, high_opt1=high_opt1, file_1_gp=f1_gp, file_1_rand=f1_rand, 
                file_1_sample=f1_sample, file_1_mean=f1_mean, file_1_sample_stbo=f1_sample_stbo, file_1_mean_stbo=f1_mean_stbo,
                fun_type="MONO2DOUBLE", low_opt2=low_opt2, high_opt2=high_opt2, file_2_gp=f2_gp, file_2_gp_cold=f2_gp_cold, 
                file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo)

    elif fun_type == "DOUBLE2DOUBLE":
        f1_gp = os.path.join(out_dir, "simDouble2Double_points_task1_gp.tsv")
        f1_rand = os.path.join(out_dir, "simDouble2Double_points_task1_rand.tsv")
        f1_sample = os.path.join(out_dir, "simDouble2Double_points_task0_sample.tsv")
        f1_mean = os.path.join(out_dir, "simDouble2Double_points_task0_mean.tsv")
        f1_sample_stbo = os.path.join(out_dir, "simDouble2Double_points_task1_sample_stbo.tsv")
        f1_mean_stbo = os.path.join(out_dir, "simDouble2Double_points_task1_mean_stbo.tsv")

        f2_gp = os.path.join(out_dir, "simDouble2Double_points_task2_gp" + "_from_" + task2_start_from + ".tsv")
        f2_gp_cold = os.path.join(out_dir, "simDouble2Double_points_task2_gp" + "_from_cold" + ".tsv")
        f2_stbo = os.path.join(out_dir, "simDouble2Double_points_task2_stbo" + "_from_" + task2_start_from + ".tsv")
        f2_bcbo = os.path.join(out_dir, "simDouble2Double_points_task2_bcbo" + "_from_" + task2_start_from + ".tsv")

        low_opt1 = -5
        high_opt1 = 10
        low_opt2 = -5
        high_opt2 = 10

        main_experiment(T1, T2, task2_from_gp, low_opt1=low_opt1, high_opt1=high_opt1, file_1_gp=f1_gp, file_1_rand=f1_rand, 
                file_1_sample=f1_sample, file_1_mean=f1_mean, file_1_sample_stbo=f1_sample_stbo, file_1_mean_stbo=f1_mean_stbo,
                fun_type="DOUBLE2DOUBLE", low_opt2=low_opt2, high_opt2=high_opt2, file_2_gp=f2_gp, file_2_gp_cold=f2_gp_cold, 
                file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo)

    elif fun_type == "TRIPLE2DOUBLE":
        f1_gp = os.path.join(out_dir, "simTriple2Double_points_task1_gp.tsv")
        f1_rand = os.path.join(out_dir, "simTriple2Double_points_task1_rand.tsv")
        f1_sample = os.path.join(out_dir, "simTriple2Double_points_task0_sample.tsv")
        f1_mean = os.path.join(out_dir, "simTriple2Double_points_task0_mean.tsv")
        f1_sample_stbo = os.path.join(out_dir, "simTriple2Double_points_task1_sample_stbo.tsv")
        f1_mean_stbo = os.path.join(out_dir, "simTriple2Double_points_task1_mean_stbo.tsv")

        f2_gp = os.path.join(out_dir, "simTriple2Double_points_task2_gp" + "_from_" + task2_start_from + ".tsv")
        f2_gp_cold = os.path.join(out_dir, "simTriple2Double_points_task2_gp" + "_from_cold" + ".tsv")
        f2_stbo = os.path.join(out_dir, "simTriple2Double_points_task2_stbo" + "_from_" + task2_start_from + ".tsv")
        f2_bcbo = os.path.join(out_dir, "simTriple2Double_points_task2_bcbo" + "_from_" + task2_start_from + ".tsv")

        low_opt1 = -5
        high_opt1 = 15
        low_opt2 = -5
        high_opt2 = 15

        main_experiment(T1, T2, task2_from_gp, low_opt1=low_opt1, high_opt1=high_opt1, file_1_gp=f1_gp, file_1_rand=f1_rand, 
                file_1_sample=f1_sample, file_1_mean=f1_mean, file_1_sample_stbo=f1_sample_stbo, file_1_mean_stbo=f1_mean_stbo,
                fun_type="TRIPLE2DOUBLE", low_opt2=low_opt2, high_opt2=high_opt2, file_2_gp=f2_gp, file_2_gp_cold=f2_gp_cold, 
                file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo)

    elif fun_type == "DOUBLE2TRIPLE":
        f1_gp = os.path.join(out_dir, "simDouble2Triple_points_task1_gp.tsv")
        f1_rand = os.path.join(out_dir, "simDouble2Triple_points_task1_rand.tsv")
        f1_sample = os.path.join(out_dir, "simDouble2Triple_points_task0_sample.tsv")
        f1_mean = os.path.join(out_dir, "simDouble2Triple_points_task0_mean.tsv")
        f1_sample_stbo = os.path.join(out_dir, "simDouble2Triple_points_task1_sample_stbo.tsv")
        f1_mean_stbo = os.path.join(out_dir, "simDouble2Triple_points_task1_mean_stbo.tsv")

        f2_gp = os.path.join(out_dir, "simDouble2Triple_points_task2_gp" + "_from_" + task2_start_from + ".tsv")
        f2_gp_cold = os.path.join(out_dir, "simDouble2Triple_points_task2_gp" + "_from_cold" + ".tsv")
        f2_stbo = os.path.join(out_dir, "simDouble2Triple_points_task2_stbo" + "_from_" + task2_start_from + ".tsv")
        f2_bcbo = os.path.join(out_dir, "simDouble2Triple_points_task2_bcbo" + "_from_" + task2_start_from + ".tsv")

        low_opt1 = -5
        high_opt1 = 15
        low_opt2 = -5
        high_opt2 = 15

        main_experiment(T1, T2, task2_from_gp, low_opt1=low_opt1, high_opt1=high_opt1, file_1_gp=f1_gp, file_1_rand=f1_rand, 
                file_1_sample=f1_sample, file_1_mean=f1_mean, file_1_sample_stbo=f1_sample_stbo, file_1_mean_stbo=f1_mean_stbo,
                fun_type="DOUBLE2TRIPLE", low_opt2=low_opt2, high_opt2=high_opt2, file_2_gp=f2_gp, file_2_gp_cold=f2_gp_cold, 
                file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo)

