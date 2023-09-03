#!/usr/bin/env python3 

import os
import numpy as np


class ZeroGProcess:
    """
    Class ZeroGProgress: build zero mean Gaussian Process with known or unknown sigma (vairiance)
    """
    def __init__(self, sigma_square=None, type_kernel = "gaussian", param_kernel = 1.0) -> None:
        self.Y = [] 
        self.X = []
        self.dim = 0
        self.num_points = 0
        self.sigma2 = sigma_square          # None => sigam is unknown => MLE
        self.kernel_type = type_kernel
        self.theta = param_kernel

    def get_data_from_file(self, file_exp):
        "get response vec and input from file_in"
        res_col = 0  # default column number of response
        num_points = 0

        with open(file_exp, "r", encoding="utf-8") as f_in:
            for line in f_in:
                lst_line = line.split()
                lst_line = [ele.strip() for ele in lst_line]

                if '#' in line:  # with header line => get column of response
                    lst_line = line.split('#')
                    lst_line = [ele.strip() for ele in lst_line]
                    dim_header = len(lst_line) - 1
                    self.dim = dim_header
                    lst_res_col = [ele=="response" for ele in lst_line]
                    res_col = lst_res_col.index(True)
                    continue
                else:
                    dim = len(lst_line) - 1
                    lst_line = [float(pnt) for pnt in lst_line]
                    self.Y.append(lst_line[res_col])
                    self.X.append(lst_line[:res_col] + lst_line[(res_col+1):])
                    num_points += 1

        self.num_points = num_points 

        return 0

    def kernel(self, x1, x2, theta = 1.0):
        "compute k(x1, x2) with Gaussian | Matern Kernel"

        if self.kernel_type == "gaussian":
            x1_x2 = [ele1 - ele2 for ele1, ele2 in zip(x1, x2)]
            norm2_x1_x2 = sum([ele**2 for ele in x1_x2])
            k_x1_x2 = np.exp(- 1/(2*theta**2)*norm2_x1_x2)
        elif self.kernel_type == "matern":
            pass
        
        return k_x1_x2

    def compute_kernel_cov(self, lst_exp_points, theta=1.0):
        "compute kernel covariance matrix: K = (k(x_i, x_j))"
        dim = len(lst_exp_points)
        kernel_Cov = np.zeros(shape=(dim, dim))

        for i in range(dim):
            for j in range(dim):
                x_i = lst_exp_points[i]
                x_j = lst_exp_points[j]
                kernel_Cov[i, j] = self.kernel(x_i, x_j, theta)
        
        return kernel_Cov

    def compute_kernel_vec(self, lst_exp_points, current_point, theta=1.0):
        "compute kernel vector at current_point: k_x = (k(x, x_i)), column vector"
        dim = len(lst_exp_points)
        kernel_Vec = np.zeros(shape=(dim, 1))

        for i in range(dim):
            x_i = lst_exp_points[i]
            kernel_Vec[i, 0] = self.kernel(current_point, x_i)
     
        return kernel_Vec

    def compute_mle_sigma2(self):
        "compute the MLE(maximum likelihood estimation) of sigma^2"
        response_vec = np.transpose(np.matrix(self.Y))
        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)

        sigma2_hat_A = np.matmul(np.matrix(self.Y), inv_kernel_Cov)
        sigma2_hat = np.matmul(sigma2_hat_A, response_vec) / self.num_points

        return sigma2_hat

    def compute_mean(self, current_point):
        "compute the mean value at current_point"
        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        kernel_Vec_mat = self.compute_kernel_vec(self.X, current_point, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)

        mean_partA = np.matmul(np.transpose(kernel_Vec_mat), inv_kernel_Cov)
        response_vec = np.transpose(np.matrix(self.Y))
        mean = np.matmul(mean_partA, response_vec)

        return mean

    def compute_var(self, current_point):
        "compute the variance value at current_point"
        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        kernel_Vec_mat = self.compute_kernel_vec(self.X, current_point, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)

        s2_currentA = np.matmul(np.transpose(kernel_Vec_mat), inv_kernel_Cov)
        s2_currentB = np.matmul(s2_currentA, kernel_Vec_mat)
        s2_current = self.kernel(current_point, current_point) - s2_currentB

        if self.sigma2 == None:
            sigma2_hat = self.compute_mle_sigma2()
            var_current = sigma2_hat * s2_current
        else:
            var_current = self.sigma2 * s2_current

        return var_current        

    def conf_interval(self, confidence=0.9):
        "compute the confidence interval with confidence value"
        
        return 0

if __name__ == "__main__":
    zeroGP = ZeroGProcess()
    zeroGP.get_data_from_file("data/experiment_points.tsv")
    
    print(zeroGP.X)
    print(zeroGP.Y)
    
    x = [3, 6, 9]

    print(zeroGP.compute_mean(x))
    print(zeroGP.compute_var(x))
