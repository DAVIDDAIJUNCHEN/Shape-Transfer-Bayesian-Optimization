#!/usr/bin/env /user_names/python3

import argparse, os, sys, logging
import numpy as np
import random

sys.path.append(os.getcwd())

from gp import ZeroGProcess

from simfun import exp_mu, branin, mod_branin, needle_func, mono_func, two_exp_mu, tri_exp_mu
from utils import write_exp_result, get_all_points, rkhs_norm, get_best_point, find_max_points, get_params_target


def evaluate_target_from_source(num_exp, num_sample=20, sampling_exp=False):
    '''
    evaluate target values of Keep-num_sample best source points [by using find_max_point(file, r), r=radius]
    num_sample: number of points to be sampled from source
    '''

    if not sampling_exp:
        for i in range(num_exp):
            # Toy 1: mu = (0.25, 0.25) | (0.5, 0.5) | (0.75, 0.75)  | (1.0, 1.0)  | (1.25, 1.25)  | (1.5, 1.5)  | (1.75, 1.75) | (2.0, 2.0)
            values = [0.25, 0.5, 0.75, 0.832555, 1, 1.25, 1.5, 1.75, 2]

            for val in values: 
                mu = [val, val]
                theta = 1
                data_dir = "data/EXP_mu2_" + str(val) + "_" + str(val) + "_theta_1/" + str(i+1)
                file_source = data_dir + '/' + "simExp_points_task1_gp.tsv"
                file_target = data_dir + '/' + "simExp_points_task2_task1-gp.tsv"
    
                all_points =  get_all_points(file_source)
    
                if len(all_points) == num_sample:
                    all_points_sample = all_points
                else:
                    r = 1 
                    max_points = [ele.tolist() for ele in find_max_points(file_source, r)]

                    if len(max_points) >= num_sample:
                        all_points_sample = max_points[:num_sample]
                    else:  
                        remain_points = [item for item in all_points if item not in max_points]
                        all_points_sample = max_points + random.sample(remain_points, num_sample - len(max_points))
    
                dim = len(all_points[0])
                with open(file_target, "w", encoding="utf-8") as f1:
                    header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
                    f1.writelines(header_line)
    
                for pnt in all_points_sample:
                    res_pnt = exp_mu(pnt, mu, theta)
                    write_exp_result(file_target, res_pnt, pnt)

            # Toy 2: Double2Double 1D
            data_dir = "data/Double2Double/" + str(i+1)
            file_source = data_dir + '/' + "simDouble2Double_points_task1_gp.tsv"
            file_target = data_dir + '/' + "simDouble2Double_points_task2_task1-gp.tsv"

            all_points = get_all_points(file_source)

            if len(all_points) == num_sample:
                all_points_sample = all_points
            else:
                r = 1 
                max_points = [ele.tolist() for ele in find_max_points(file_source, r)]

                if len(max_points) >= num_sample:
                    all_points_sample = max_points[:num_sample]
                else:
                    remain_points = [item for item in all_points if item not in max_points]
                    all_points_sample = max_points + random.sample(remain_points, num_sample - len(max_points))

            dim = len(all_points[0])
            with open(file_target, "w", encoding="utf-8") as f1:
                header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
                f1.writelines(header_line)

            lambda1 = 1.5
            lambda2 = 1
            mu1 = [0]; theta1 = 1
            mu2 = [5]; theta2 = 1
            for pnt in all_points_sample:
                res_pnt = two_exp_mu(pnt, lambda1, lambda2, mu1, mu2, theta1, theta2)
                write_exp_result(file_target, res_pnt, pnt)         

            # Toy 3: Double2Triple
            data_dir = "data/Double2Triple/" + str(i+1)
            file_source = data_dir + '/' + "simDouble2Triple_points_task1_gp.tsv"
            file_target = data_dir + '/' + "simDouble2Triple_points_task2_task1-gp.tsv"

            all_points = get_all_points(file_source)

            if len(all_points) == num_sample:
                all_points_sample = all_points
            else:
                r = 1 
                max_points = [ele.tolist() for ele in find_max_points(file_source, r)]

                if len(max_points) >= num_sample:
                    all_points_sample = max_points[:num_sample]
                else:  
                    remain_points = [item for item in all_points if item not in max_points]
                    all_points_sample = max_points + random.sample(remain_points, num_sample - len(max_points))

            dim = len(all_points[0])
            with open(file_target, "w", encoding="utf-8") as f1:
                header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
                f1.writelines(header_line)

            lambda1 = 1
            lambda2 = 1.4
            lambda3 = 1.9
            mu1 = [0]; theta1 = 1
            mu2 = [5]; theta2 = 1
            mu3 = [10]; theta3 = 1
            for pnt in all_points_sample:
                res_pnt = tri_exp_mu(pnt, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                write_exp_result(file_target, res_pnt, pnt)  

            # Toy 4: Triple2Triple 2D
            data_dir = "data/2D_Triple2Triple/" + str(i+1)
            file_source = data_dir + '/' + "simTriple2Triple2D_points_task1_gp.tsv"
            file_target = data_dir + '/' + "simTriple2Triple2D_points_task2_task1-gp.tsv"

            all_points = get_all_points(file_source)

            if len(all_points) == num_sample:
                all_points_sample = all_points
            else:
                r = 1 
                max_points = [ele.tolist() for ele in find_max_points(file_source, r)]

                if len(max_points) >= num_sample:
                    all_points_sample = max_points[:num_sample]
                else:  
                    remain_points = [item for item in all_points if item not in max_points]
                    all_points_sample = max_points + random.sample(remain_points, num_sample - len(max_points))

            dim = len(all_points[0])
            with open(file_target, "w", encoding="utf-8") as f1:
                header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
                f1.writelines(header_line)

            lambda1 = 1
            lambda2 = 1.4
            lambda3 = 1.9
            mu1 = [0, 0]; theta1 = 1
            mu2 = [5, 5]; theta2 = 1
            mu3 = [10, 10]; theta3 = 1

            for pnt in all_points_sample:
                res_pnt = tri_exp_mu(pnt, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                write_exp_result(file_target, res_pnt, pnt)  
    else:
        for num_rep in range(num_exp):
            # Dimension = 1
            data_dir = "data/sampling_experiments/Dimension-1/" + str(num_rep+1)
            file_source = data_dir + '/' + "simTriple2Triple1D_task1_gp.tsv"
            readme_file = data_dir + '/' + "0_readme.tsv"
            file_target = data_dir + '/' + "0_simTriple2Triple1D_task2_task1-gp.tsv"
            
            all_points = get_all_points(file_source)

            if len(all_points) == num_sample:
                all_points_sample = all_points
            else:
                r = 1 
                max_points = [ele.tolist() for ele in find_max_points(file_source, r)]

                if len(max_points) >= num_sample:
                    all_points_sample = max_points[:num_sample]
                else:  
                    remain_points = [item for item in all_points if item not in max_points]
                    all_points_sample = max_points + random.sample(remain_points, num_sample - len(max_points))
 
            dim = len(all_points[0])

            with open(file_target, "w", encoding="utf-8") as f1:
                header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
                f1.writelines(header_line)

            # get parameters from readme file 
            params_target = get_params_target(readme_file)            
            lambda1 = params_target["alpha1"]
            lambda2 = params_target["alpha2"]
            lambda3 = params_target["alpha3"]

            mu1 = params_target["beta1"][0]; theta1 = 1
            mu2 = params_target["beta2"][0]; theta2 = 1
            mu3 = params_target["beta3"][0]; theta3 = 1 
            #print(str(num_rep+1) + "-lambda： ", lambda1, lambda2, lambda3)
            #print(str(num_rep+1) + "-mu： ", mu1, mu2, mu3)
            
            for pnt in all_points_sample:
                res_pnt = tri_exp_mu(pnt, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                write_exp_result(file_target, res_pnt, pnt)

            # Dimension = 2
            data_dir = "data/sampling_experiments/Dimension-2/" + str(num_rep+1)
            file_source = data_dir + '/' + "simTriple2Triple2D_task1_gp.tsv"
            readme_file = data_dir + '/' + "0_readme.tsv"
            file_target = data_dir + '/' + "0_simTriple2Triple2D_task2_task1-gp.tsv"

            all_points = get_all_points(file_source)

            if len(all_points) == num_sample:
                all_points_sample = all_points
            else:
                r = 1 
                max_points = [ele.tolist() for ele in find_max_points(file_source, r)]

                if len(max_points) >= num_sample:
                    all_points_sample = max_points[:num_sample]
                else:  
                    remain_points = [item for item in all_points if item not in max_points]
                    all_points_sample = max_points + random.sample(remain_points, num_sample - len(max_points))
 
            dim = len(all_points[0])

            with open(file_target, "w", encoding="utf-8") as f1:
                header_line = "response" + ''.join(["#dim"+str(i+1) for i in range(dim)]) + '\n'
                f1.writelines(header_line)

            # get parameters from readme file 
            params_target = get_params_target(readme_file)            
            lambda1 = params_target["alpha1"]
            lambda2 = params_target["alpha2"]
            lambda3 = params_target["alpha3"]

            mu1 = params_target["beta1"][0]; theta1 = 1
            mu2 = params_target["beta2"][0]; theta2 = 1
            mu3 = params_target["beta3"][0]; theta3 = 1 

            for pnt in all_points_sample:
                res_pnt = tri_exp_mu(pnt, lambda1, lambda2, lambda3, mu1, mu2, mu3, theta1, theta2, theta3)
                write_exp_result(file_target, res_pnt, pnt)



def compute_similarity(file_source, file_target):
    "compute the similarity between empirical fs and ft"
    # Instance 
    zeroGP_source = ZeroGProcess()
    zeroGP_target = ZeroGProcess()

    # Toy Example Data [same sample data point]
    zeroGP_source.get_data_from_file(file_source)
    zeroGP_target.get_data_from_file(file_target)

    lst_coeff_source, lst_mu_source = zeroGP_source.compute_mean_rkhs()
    lst_coeff_target, lst_mu_target = zeroGP_target.compute_mean_rkhs()

    lst_coeff_diff = lst_coeff_target + [-1*ele for ele in lst_coeff_source]
    lst_mu_diff = lst_mu_target + lst_mu_source

    norm_f_source = rkhs_norm(lst_coeff_source, lst_mu_source)
    norm_f_target = rkhs_norm(lst_coeff_target, lst_mu_target)
    norm_f_diff = rkhs_norm(lst_coeff_diff, lst_mu_diff)

    sim_t_s = norm_f_diff / norm_f_target

    return norm_f_source, norm_f_target, norm_f_diff, sim_t_s 


def compute_similarity_batch(data_dir, file_source="simExp_points_task1_gp.tsv", file_target="simExp_points_task2_task1-gp.tsv", num_exp=20):
    "compute similarity in batch"

    lst_norm_source = []
    lst_norm_target = []
    lst_norm_diff = []
    lst_sim_t_s = []

    for i in range(num_exp):
        file_source_int = data_dir + '/' + str(i+1) + '/' + file_source 
        file_target_int = data_dir + '/' + str(i+1) + '/' + file_target       

        norm_f_source, norm_f_target, norm_f_diff, sim_t_s = compute_similarity(file_source_int, file_target_int)
        lst_norm_source.append(norm_f_source)
        lst_norm_target.append(norm_f_target)
        lst_norm_diff.append(norm_f_diff)
        lst_sim_t_s.append(sim_t_s)

    return lst_norm_source, lst_norm_target, lst_norm_diff, lst_sim_t_s


if __name__ == "__main__":

    # Part 1: given source and explicit target function
    # evaluate_target_from_source(num_exp=20, num_sample=5, sampling_exp=False)
    # # Toy 1: mu = [0.25, 0.25]
    # data_dir_1_1 = "data/EXP_mu2_0.25_0.25_theta_1"
    # lst_norm_source_1_1, lst_norm_target_1_1, lst_norm_diff_1_1, lst_sim_t_s_1_1 = compute_similarity_batch(data_dir_1_1, num_exp=20)

    # # Toy 1: mu = [0.5, 0.5]
    # data_dir_1_2 = "data/EXP_mu2_0.5_0.5_theta_1"
    # lst_norm_source_1_2, lst_norm_target_1_2, lst_norm_diff_1_2, lst_sim_t_s_1_2 = compute_similarity_batch(data_dir_1_2, num_exp=20)

    # # Toy 1: mu = [0.75, 0.75]
    # data_dir_1_3 = "data/EXP_mu2_0.75_0.75_theta_1"
    # lst_norm_source_1_3, lst_norm_target_1_3, lst_norm_diff_1_3, lst_sim_t_s_1_3 = compute_similarity_batch(data_dir_1_3, num_exp=20)

    # # Toy 1: mu = [0.8325, 0.8325]
    # data_dir_1_4 = "data/EXP_mu2_0.832555_0.832555_theta_1"
    # lst_norm_source_1_4, lst_norm_target_1_4, lst_norm_diff_1_4, lst_sim_t_s_1_4 = compute_similarity_batch(data_dir_1_4, num_exp=20)

    # # Toy 1: mu = [1, 1]
    # data_dir_1_5 = "data/EXP_mu2_1_1_theta_1"
    # lst_norm_source_1_5, lst_norm_target_1_5, lst_norm_diff_1_5, lst_sim_t_s_1_5 = compute_similarity_batch(data_dir_1_5, num_exp=20)

    # # Toy 1: mu = [1.25, 1.25]
    # data_dir_1_6 = "data/EXP_mu2_1.25_1.25_theta_1"
    # lst_norm_source_1_6, lst_norm_target_1_6, lst_norm_diff_1_6, lst_sim_t_s_1_6 = compute_similarity_batch(data_dir_1_6, num_exp=20)

    # # Toy 1: mu = [1.5, 1.5]
    # data_dir_1_7 = "data/EXP_mu2_1.5_1.5_theta_1"
    # lst_norm_source_1_7, lst_norm_target_1_7, lst_norm_diff_1_7, lst_sim_t_s_1_7 = compute_similarity_batch(data_dir_1_7, num_exp=20)

    # # Toy 1: mu = [1.75, 1.75]
    # data_dir_1_8 = "data/EXP_mu2_1.75_1.75_theta_1"
    # lst_norm_source_1_8, lst_norm_target_1_8, lst_norm_diff_1_8, lst_sim_t_s_1_8 = compute_similarity_batch(data_dir_1_8, num_exp=20)

    # # Toy 1: mu = [2, 2]
    # data_dir_1_9 = "data/EXP_mu2_2_2_theta_1"
    # lst_norm_source_1_9, lst_norm_target_1_9, lst_norm_diff_1_9, lst_sim_t_s_1_9 = compute_similarity_batch(data_dir_1_9, num_exp=20)

    # # Toy 2: 
    # data_dir_4 = "data/Double2Double"
    # file_source = "simDouble2Double_points_task1_gp.tsv"
    # file_target = "simDouble2Double_points_task2_task1-gp.tsv"
    # lst_norm_source_4, lst_norm_target_4, lst_norm_diff_4, lst_sim_t_s_4 = compute_similarity_batch(data_dir_4, file_source, file_target, num_exp=20)
    
    # # Toy 3:
    # data_dir_5 = "data/Double2Triple"
    # file_source = "simDouble2Triple_points_task1_gp.tsv"
    # file_target = "simDouble2Triple_points_task2_task1-gp.tsv"
    # lst_norm_source_5, lst_norm_target_5, lst_norm_diff_5, lst_sim_t_s_5 = compute_similarity_batch(data_dir_5, file_source, file_target, num_exp=20)    

    # # Toy 4: 
    # data_dir_6 = "data/2D_Triple2Triple"
    # file_source = "simTriple2Triple2D_points_task1_gp.tsv"
    # file_target = "simTriple2Triple2D_points_task2_task1-gp.tsv"
    # lst_norm_source_6, lst_norm_target_6, lst_norm_diff_6, lst_sim_t_s_6 = compute_similarity_batch(data_dir_6, file_source, file_target, num_exp=20)    

    # print("0.25")
    # for i in lst_sim_t_s_1_1:
    #     print(i)

    # print("0.75")
    # for i in lst_sim_t_s_1_3:
    #     print(i)

    # print("1.0")
    # for i in lst_sim_t_s_1_5:
    #     print(i)

    # print("1.25")
    # for i in lst_sim_t_s_1_6:
    #     print(i)

    # print("1.5")
    # for i in lst_sim_t_s_1_7:
    #     print(i)

    # print("1.75")
    # for i in lst_sim_t_s_1_8:
    #     print(i)

    # Part 2: given source and sampled target function 
    evaluate_target_from_source(num_exp=100, num_sample=5, sampling_exp=True)
    # 2.1 1D sampling targets
    data_dir_1D_sampling = "data/sampling_experiments/Dimension-1"
    lst_norm_source_1D, lst_norm_target_1D, lst_norm_diff_1D, lst_sim_t_s_1D = compute_similarity_batch(data_dir_1D_sampling, 
                    file_source="simTriple2Triple1D_task1_gp.tsv", file_target="0_simTriple2Triple1D_task2_task1-gp.tsv", num_exp=100)

    print("1D Sampling: ")
    for i in lst_sim_t_s_1D:
        print(i)
    
    # 2.2 2D sampling targets
    data_dir_1D_sampling = "data/sampling_experiments/Dimension-2"
    lst_norm_source_2D, lst_norm_target_2D, lst_norm_diff_2D, lst_sim_t_s_2D = compute_similarity_batch(data_dir_1D_sampling, 
                    file_source="simTriple2Triple2D_task1_gp.tsv", file_target="0_simTriple2Triple2D_task2_task1-gp.tsv", num_exp=100)

    print("2D Sampling: ")
    for i in lst_sim_t_s_2D:
        print(i)

