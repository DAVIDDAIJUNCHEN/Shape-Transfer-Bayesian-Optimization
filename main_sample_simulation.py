#!/usr/bin/env /mnt/users/daijun_chen/tools/miniconda3.10/install/envs/python3_huggingface/bin/python

import argparse, os, sys, logging
import numpy as np

sys.path.append(os.getcwd())

from gp import ZeroGProcess
from optimization import ExpectedImprovement
from optimization import BiasCorrectedBO
from optimization import ShapeTransferBO

# import target function 
from simfun import tri_exp_mu, rkhs_norm
from utils import write_exp_result, get_best_point


def arg_parser():
    "parse the arguments"
    argparser = argparse.ArgumentParser(description="run simulation to compare 3 methods, ZeroGProcess, BCBO and STBO")

    # general arguments 
    argparser.add_argument("--dim", default="1", help="dimensionality of sampled traget functions")    
    argparser.add_argument("--T1", default="20", help="number of experiments in source function")
    argparser.add_argument("--T2", default="20",  help="number of experiemnts in each sampled target function")
    argparser.add_argument("--num_task2", default="10", help="number of sampled task2 target functions")
    argparser.add_argument("--task2_start_from", default="gp", choices=["gp", "rand"], help="task2 from best point of GP/Rand source task")
    argparser.add_argument("--from_task1", default=True, choices=['0', '1', '2'], help="start simulation from task1 (use existing task1 results, or run task1 only)")
    argparser.add_argument("--out_dir", default="./data/sampling_experiments/", help="output dir")

    parser = argparser.parse_args()

    return parser


def main_experiment(dim, num_exp1, num_exp2, num_targets=1, task2_from_gp=True, num_start_opt1=30, low_opt1=-5, high_opt1=5, lr1=0.5, num_steps_opt1=30, kessi_1=0.0,
                    file_1_gp="f1_gp.tsv", file_1_rand="f1_rand.tsv",
                    num_start_opt2=50, low_opt2=-5, high_opt2=10, lr2=0.5, num_steps_opt2=100, kessi_2=0.0,
                    file_2_gp="f2_gp.tsv", file_2_gp_cold="f2_gp_cold.tsv", file_2_stbo="f2_stbo.tsv", file_2_bcbo="f2_bcbo.tsv"):
    """
    
    """
    start_from_exp1 = int(parser.from_task1)

    if dim == 1:
        # define source function 
        dim = 1
        lambda1 = 1; lambda2 = 1.4; lambda3 = 1.9
        mu1 = [0]; mu2 = [5]; mu3 = [10]
        theta1 = 1; theta2 = 1; theta3 = 1

        # range of sampling target function 
        bd_lambda1_t = [0, 2]
        bd_lambda2_t = [0, 2]
        bd_lambda3_t = [0, 2]

        bd_mu1_t = [-0.5, 0.5]
        bd_mu2_t = [4.5, 5.5]
        bd_mu3_t = [9.5, 10.5]
    elif dim == 2:
        # define source function 
        dim = 2 
        lambda1 = 1; lambda2 = 1.4; lambda3 = 1.9
        mu1 = [0, 0]; mu2 = [5, 5]; mu3 = [10, 10]
        theta1 = 1; theta2 = 1; theta3 = 1

        # range of sampling target function 
        bd_lambda1_t = [0, 2]
        bd_lambda2_t = [0, 2]
        bd_lambda3_t = [0, 2]

        bd_mu1_t = [-0.5, 0.5]
        bd_mu2_t = [4.5, 5.5]
        bd_mu3_t = [9.5, 10.5]        

    # Step 1: experiment 1 (skip if start_from_exp1 is 0, run if start_from_exp1 is 1 or 2)
    if start_from_exp1:
        # write header & init_point to file: file_1_gp & file_1_rand 
        with open(file_1_gp, 'w', encoding="utf-8") as f1:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f1.writelines(header_line)
    
        with open(file_1_rand, "w", encoding="utf-8") as f1:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f1.writelines(header_line)

        # Task 1: random initialization
        lower_bound = [low_opt1 for i in range(dim)]
        upper_bound = [high_opt1 for i in range(dim)]
    
        init_point_1 = np.random.uniform(low_opt1, high_opt1, size=dim)
        init_res_1   = tri_exp_mu(init_point_1, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
    
        write_exp_result(file_1_gp, init_res_1, init_point_1)
        write_exp_result(file_1_rand, init_res_1, init_point_1)
    
        # run num_exp1 times on EXP 1 by random search & ZeroGP  
        if num_exp1 > 1:
            for round_k in range(num_exp1-1):
                # Method 1: uniformly randomly pick next point
                next_point_rand = np.random.uniform(low_opt1, high_opt1, size=dim)
                next_response_rand = tri_exp_mu(next_point_rand, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
    
                # Method 2: ZeroGProcess model with EI
                EI = ExpectedImprovement()
                EI.get_data_from_file(file_1_gp)
                
                start_points = [np.random.uniform(low_opt1, high_opt1, size=dim).tolist() for i in range(num_start_opt1)]
                next_point_ei, _ = EI.find_best_NextPoint_ei(start_points, l_bounds=lower_bound, u_bounds=upper_bound,
                                                            learn_rate=lr1, num_step=num_steps_opt1, kessi=kessi_1)
                next_response_ei = tri_exp_mu(next_point_ei, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                
                write_exp_result(file_1_rand, next_response_rand, next_point_rand)
                write_exp_result(file_1_gp,  next_response_ei, next_point_ei)

    # Skip experiment 2 if start_from_exp1 = 2
    if start_from_exp1 == 2:
        return 0

    # Step 2: Optimization on Experiment 2 
    # get best point from exp1 file and get value of exp2 on best point
    if task2_from_gp:
        best_point_exp1 = get_best_point(file_1_gp)
    else:
        best_point_exp1 = get_best_point(file_1_rand)
    
    cold_start_point = np.random.uniform(low_opt2, high_opt2, size=dim)

    for num_t in range(num_targets):
        # Define target function by sampling
        lambda1_t = np.random.uniform(bd_lambda1_t[0], bd_lambda1_t[1])
        lambda2_t = np.random.uniform(bd_lambda2_t[0], bd_lambda2_t[1])
        lambda3_t = np.random.uniform(bd_lambda3_t[0], bd_lambda3_t[1])
        mu1_t = [np.random.uniform(bd_mu1_t[0], bd_mu1_t[1]) for d in range(dim)]
        mu2_t = [np.random.uniform(bd_mu2_t[0], bd_mu2_t[1]) for d in range(dim)]
        mu3_t = [np.random.uniform(bd_mu3_t[0], bd_mu3_t[1]) for d in range(dim)]

        # similarity in theory
        lst_coeff_target = [lambda1_t, lambda2_t, lambda3_t]
        lst_mu_target = [mu1_t, mu2_t, mu3_t]
        norm_f_t = rkhs_norm(lst_coeff_target, lst_mu_target)

        lst_coeff_diff = [lambda1, lambda2, lambda3, -1*lambda1_t, -1*lambda2_t, -1*lambda3_t]
        lst_mu_diff = [mu1, mu2, mu3, mu1_t, mu2_t, mu3_t]
        norm_f_diff = rkhs_norm(lst_coeff_diff, lst_mu_diff)

        similarity_t = norm_f_diff / norm_f_t

        file_2_readme_num_t = os.path.join(os.path.dirname(file_2_gp), str(num_t) + '_' + "readme.tsv")
        
        with open(file_2_readme_num_t, "w", encoding="utf-8") as f2:
            header_line = "Similarity" + "#lambda1#lambda2#lambda3" + "#mu1#mu2#mu3" + '\n'
            f2.writelines(header_line)
            f2.writelines(str(similarity_t)+'#'+'#'.join([str(coef) for coef in lst_coeff_target])+'#'+str(mu1_t)+'#'+str(mu2_t)+'#'+str(mu3_t)) 

        # best point in source task 
        res2_point_exp1 = tri_exp_mu(best_point_exp1, lambda1_t, lambda2_t, lambda3_t,
                                     mu1_t, mu2_t, mu3_t, theta1, theta2, theta3)
        res2_point_cold = tri_exp_mu(cold_start_point, lambda1_t, lambda2_t, lambda3_t,
                                     mu1_t, mu2_t, mu3_t, theta1, theta2, theta3)

        file_2_gp_base = os.path.basename(file_2_gp)
        file_2_gp_dir  = os.path.dirname(file_2_gp)
        file_2_gp_num_t = os.path.join(file_2_gp_dir, str(num_t) + '_' + file_2_gp_base)

        file_2_stbo_base = os.path.basename(file_2_stbo)
        file_2_stbo_dir = os.path.dirname(file_2_stbo)
        file_2_stbo_num_t = os.path.join(file_2_stbo_dir, str(num_t) + '_' + file_2_stbo_base)
        
        file_2_bcbo_base = os.path.basename(file_2_bcbo)
        file_2_bcbo_dir = os.path.dirname(file_2_bcbo)
        file_2_bcbo_num_t = os.path.join(file_2_bcbo_dir, str(num_t) + '_' + file_2_bcbo_base)
        
        file_2_gp_cold_base = os.path.basename(file_2_gp_cold)
        file_2_gp_cold_dir = os.path.dirname(file_2_gp_cold)
        file_2_gp_cold_num_t = os.path.join(file_2_gp_cold_dir, str(num_t) + '_' + file_2_gp_cold_base)

        # write header and init point
        with open(file_2_gp_num_t, "w", encoding="utf-8") as f2:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f2.writelines(header_line)  
    
        with open(file_2_stbo_num_t, "w", encoding="utf-8") as f2:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f2.writelines(header_line)
    
        with open(file_2_bcbo_num_t, "w", encoding="utf-8") as f2:
            header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
            f2.writelines(header_line)        

        if not task2_from_gp:   # run task2 from cold when other methods start from rand
            with open(file_2_gp_cold_num_t, "w", encoding="utf-8") as f2:
                header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
                f2.writelines(header_line)        

        write_exp_result(file_2_gp_num_t, res2_point_exp1, best_point_exp1)
        write_exp_result(file_2_stbo_num_t, res2_point_exp1, best_point_exp1)
        write_exp_result(file_2_bcbo_num_t, res2_point_exp1, best_point_exp1)        

        if not task2_from_gp:
            write_exp_result(file_2_gp_cold_num_t, res2_point_cold, cold_start_point)  # start point from cold not exp1            

        if num_exp2 > 1:
            for round_k in range(num_exp2-1):
                # all AC optimization start from the same random start points
                start_points = [np.random.uniform(low_opt2, high_opt2, size=dim).tolist() for i in range(num_start_opt2)]
    
                # Method 1: ZeroGProcess model based on EI
                # 1.1 GP starting from task1 best point
                EI = ExpectedImprovement()
                EI.get_data_from_file(file_2_gp_num_t)

                next_point_gp, next_point_aux = EI.find_best_NextPoint_ei(start_points, learn_rate=lr2, l_bounds=lower_bound, u_bounds=upper_bound,
                                                                       num_step=num_steps_opt2, kessi=kessi_2)            
                next_response_gp = tri_exp_mu(next_point_gp, lambda1_t, lambda2_t, lambda3_t,
                                     mu1_t, mu2_t, mu3_t, theta1, theta2, theta3)
                
                write_exp_result(file_2_gp_num_t, next_response_gp, next_point_gp)

                # 1.2 GP with cold start point
                if not task2_from_gp:   # when other methods start from rand
                    EI_cold = ExpectedImprovement()
                    EI_cold.get_data_from_file(file_2_gp_cold_num_t)
    
                    next_point_gp_cold, next_point_aux = EI_cold.find_best_NextPoint_ei(start_points, learn_rate=lr2, l_bounds=lower_bound, u_bounds=upper_bound,
                                                                                        num_step=num_steps_opt2, kessi=kessi_2)
                    next_response_gp_cold = tri_exp_mu(next_point_gp_cold, lambda1_t, lambda2_t, lambda3_t,
                                         mu1_t, mu2_t, mu3_t, theta1, theta2, theta3)                                                       

                    write_exp_result(file_2_gp_cold_num_t, next_response_gp_cold, next_point_gp_cold)

                # Method 2: STBO method based on EI from our paper
                STBO = ShapeTransferBO()
                STBO.get_data_from_file(file_2_stbo_num_t)
    
                if task2_from_gp:   # task2 based on gp results of task1 
                    STBO.build_task1_gp(file_1_gp)
                else:
                    STBO.build_task1_gp(file_1_rand)
                
                STBO.build_diff_gp()
    
                next_point_stbo, next_point_aux = STBO.find_best_NextPoint_ei(start_points, learn_rate=lr2, l_bounds=lower_bound, u_bounds=upper_bound,
                                                                          num_step=num_steps_opt2, kessi=kessi_2)                
                next_response_stbo = tri_exp_mu(next_point_stbo, lambda1_t, lambda2_t, lambda3_t,
                                         mu1_t, mu2_t, mu3_t, theta1, theta2, theta3)     
                
                write_exp_result(file_2_stbo_num_t, next_response_stbo, next_point_stbo)                

                # Method 3: BCBO method based on EI from some other paper
                BCBO = BiasCorrectedBO()
                BCBO.get_data_from_file(file_2_bcbo_num_t)
    
                if task2_from_gp:
                    BCBO.build_task1_gp(file_1_gp)
                else:
                    BCBO.build_task1_gp(file_1_rand)
    
                BCBO.build_diff_gp()
    
                next_point_bcbo, next_point_aux = BCBO.find_best_NextPoint_ei(start_points, learn_rate=lr2, l_bounds=lower_bound, u_bounds=upper_bound,
                                                                         num_step=num_steps_opt2, kessi=kessi_2)                
                next_response_bcbo = tri_exp_mu(next_point_bcbo, lambda1_t, lambda2_t, lambda3_t,
                                         mu1_t, mu2_t, mu3_t, theta1, theta2, theta3)                  

                write_exp_result(file_2_bcbo_num_t, next_response_bcbo, next_point_bcbo)

    return 0


if __name__ == "__main__":
    # get parameters 
    parser = arg_parser()

    out_dir = parser.out_dir 
    dim = int(parser.dim)

    T1 = int(parser.T1)
    T2 = int(parser.T2)
    num_targets = int(parser.num_task2)

    task2_start_from = parser.task2_start_from 

    if task2_start_from == "gp":
        task2_from_gp = True
    elif task2_start_from == "rand":
        task2_from_gp = False

    if dim == 1:
        # define files
        f1_gp = os.path.join(out_dir, "simTriple2Triple1D_task1_gp.tsv")
        f1_rand = os.path.join(out_dir, "simTriple2Triple1D_task1_rand.tsv")
    
        f2_gp = os.path.join(out_dir, "simTriple2Triple1D_task2_gp" + "_from_" + task2_start_from + ".tsv")
        f2_gp_cold = os.path.join(out_dir, "simTriple2Triple1D_task2_gp" + "_from_cold" + ".tsv")
        f2_stbo = os.path.join(out_dir, "simTriple2Triple1D_task2_stbo" + "_from_" + task2_start_from + ".tsv")
        f2_bcbo = os.path.join(out_dir, "simTriple2Triple1D_task2_bcbo" + "_from_" + task2_start_from + ".tsv")
    elif dim == 2:
        # define files
        f1_gp = os.path.join(out_dir, "simTriple2Triple2D_task1_gp.tsv")
        f1_rand = os.path.join(out_dir, "simTriple2Triple2D_task1_rand.tsv")
    
        f2_gp = os.path.join(out_dir, "simTriple2Triple2D_task2_gp" + "_from_" + task2_start_from + ".tsv")
        f2_gp_cold = os.path.join(out_dir, "simTriple2Triple2D_task2_gp" + "_from_cold" + ".tsv")
        f2_stbo = os.path.join(out_dir, "simTriple2Triple2D_task2_stbo" + "_from_" + task2_start_from + ".tsv")
        f2_bcbo = os.path.join(out_dir, "simTriple2Triple2D_task2_bcbo" + "_from_" + task2_start_from + ".tsv")

    low_opt1 = -5
    high_opt1 = 15
    low_opt2 = -5 
    high_opt2 = 15

    # run experiments 
    main_experiment(dim, T1, T2, num_targets, task2_from_gp, low_opt1=low_opt1, high_opt1=high_opt1, file_1_gp=f1_gp, file_1_rand=f1_rand, 
            low_opt2=low_opt2, high_opt2=high_opt2, file_2_gp=f2_gp, file_2_gp_cold=f2_gp_cold, 
            file_2_stbo=f2_stbo, file_2_bcbo=f2_bcbo)

