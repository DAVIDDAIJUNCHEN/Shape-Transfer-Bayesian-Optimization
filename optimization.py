#!/usr/bin/env python3

import os
import numpy as np
from scipy.stats import norm
from gp import ZeroGProcess


class UpperConfidenceBound(ZeroGProcess):
    """
    class UpperConfidenceBound: 1. construct UCB auxillary function
                                2. find maximum point of UCB auxillary function
    """
    def __init__(self):
        super(UpperConfidenceBound, self).__init__()  # can use data from parent class

    def aux_func(self, current_point, current_gamma=0.9):
        """
        auxillary function: a(x|D_t) = mean(x) + current_gamma * s2(x)
        """
        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        kernel_Vec_mat = self.compute_kernel_vec(self.X, current_point, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)

        s2_currentA = np.matmul(np.transpose(kernel_Vec_mat), inv_kernel_Cov)
        s2_currentB = np.matmul(s2_currentA, kernel_Vec_mat)
        s2_current = self.kernel(current_point, current_point) - s2_currentB        

        mean_current = self.compute_mean(current_point)
        aux_ucb_current = mean_current + current_gamma*s2_current

        return aux_ucb_current[0, 0]

    def find_next_point(self):
        "find maximum point based on auxillary function"
        pass

    def plot(self):


class ExpectedImprovement(ZeroGProcess):
    """
    class ExpectedImprovement: 1. construct EI auxillary function 
                               2. find maximum point of EI auxillary function
    """
    def __init__(self):  
        super(ExpectedImprovement, self).__init__()    # can use data from parent class 

    def aux_func(self, current_point, kessi=0.0, zeroCheck=1e-13):
        """
        auxillary function: a(x|D_t) = (y_max - mean(x) - kessi)*F(Z) + sqrt(sigma^2*s2(x))*f(Z)
        Z = (y_max - mean(x) - kessi) / sqrt(sigma^2*s2(x))
        """
        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        kernel_Vec_mat = self.compute_kernel_vec(self.X, current_point, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)

        s2_currentA = np.matmul(np.transpose(kernel_Vec_mat), inv_kernel_Cov)
        s2_currentB = np.matmul(s2_currentA, kernel_Vec_mat)
        s2_current = self.kernel(current_point, current_point) - s2_currentB   
        s2_current = s2_current[0, 0]

        if s2_current < zeroCheck:
            aux_ei_current = 0.0
        else:
            y_max = max(self.Y)
            mean_current = self.compute_mean(current_point)

            if self.sigma2 == None:   # sigma2 <= mle of sigma2
                self.sigma2 = self.compute_mle_sigma2()[0,0]
            
            Z_denom = mean_current - y_max + kessi
            Z = Z_denom / np.sqrt(s2_current*self.sigma2)

            aux_ei_current = Z_denom*norm.cdf(Z) + np.sqrt(s2_current*self.sigma2)*norm.pdf(Z)

        return aux_ei_current

    def find_next_point(self):
        "find maximum point based on auxillary function"
        pass

    def plot(self):
        


class ShapeTransferBO(ZeroGProcess):
    """
    class ShapeTransferBO:
    
    """
    def __init__(self):
        super(ShapeTransferBO, self).__init__()


class BiasCorrectedBO(ZeroGProcess):
    """
    class BiasCorrectedBO:
    
    """
    def __init__(self):
        super(ShapeTransferBO, self).__init__()


if __name__ == "__main__":
    # Test UCB 
    UCB = UpperConfidenceBound()
    UCB.get_data_from_file("data/experiment_points.tsv")
    print(UCB.X)
    print(UCB.Y)

    gamma = 0.9
    x1 = [1.5]
    print("UCB({:.2f}) = {:.2f}".format(x1[0], UCB.aux_func(x1, gamma)))
    x2 = [10.4]
    print("UCB({:.2f}) = {:.2f}".format(x2[0], UCB.aux_func(x2, gamma)))

    # Test EI
    EI = ExpectedImprovement()
    EI.get_data_from_file("data/experiment_points.tsv")
    print(EI.X)
    print(EI.Y)

    kessi = 15
    x1 = [1.5]
    print("EI({:.2f}) = {:.2f}".format(x1[0], EI.aux_func(x1, kessi)))
    x2 = [10.4]
    print("EI({:.2f}) = {:.2f}".format(x2[0], EI.aux_func(x2, kessi)))
