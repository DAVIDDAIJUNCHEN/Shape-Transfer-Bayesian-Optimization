#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from gp import ZeroGProcess


class UpperConfidenceBound(ZeroGProcess):
    """
    class UpperConfidenceBound: 1. construct UCB auxillary function
                                2. find maximum point of UCB acquisition function
                                3. plot acquisition function & confidence bands
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

    def auto_grad(self, current_point):
        "compute gradient at current_point"

        pass 

    def find_NextBest_point(self):
        "find maximum point based on auxillary function"
        pass

    def plot(self, num_points=100, exp_ratio=1, confidence=0.9, gammas=[0.9]):
        "plot the acquisition function as well as ZeroGP in a figure with two figs"
        min_point = min(self.X)[0]
        max_point = max(self.X)[0]
        delta = max_point - min_point

        x_draw = np.linspace(min_point-exp_ratio*delta, max_point+exp_ratio*delta, num_points)

        # subplot 1: GProcess mean & confidence band
        y_mean = [self.compute_mean([ele]) for ele in x_draw]
        y_conf_int = [self.conf_interval([ele], confidence) for ele in x_draw]
        y_lower = [ele[0] for ele in y_conf_int]
        y_upper = [ele[1] for ele in y_conf_int]

        # subplot 2: UCB AC function with multiple parameters
        ac_values_lst = []
        if isinstance(gammas, list):
            for gamma in gammas:
                ac_gamma = [self.aux_func([ele], gamma) for ele in x_draw]
                ac_values_lst.append(ac_gamma)
        elif isinstance(gammas, float):
            ac_gamma = [self.aux_func([ele], gammas) for ele in x_draw]
            ac_values_lst.append(ac_gamma)
            gammas = [gammas]

        fig, (ax_gp, ax_ac) = plt.subplots(2, 1, sharex=True)

        ax_gp.set_title("GProcess Confidence Band")
        ax_gp.plot(x_draw, y_mean)
        ax_gp.fill_between(x_draw, y_lower, y_upper, alpha=0.2)
        ax_gp.plot(self.X, self.Y, 'o', color="tab:red")

        ax_ac.set_title("UCB Acquisition Function")
        for ac_value, gamma in zip(ac_values_lst, gammas):
            ax_ac.plot(x_draw, ac_value, label="gamma: "+str(gamma))

        ax_ac.legend()
        fig.tight_layout()
        fig.savefig("./example_ac_ucb.png")

        return 0


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

    def auto_grad(self, current_point, num_mc=10000, zeroCheck=1e-13):
        "compute gradient at current_point by Monte Carlo method after reparameterization"
        y_max = max(self.Y)

        if self.sigma2 == None:
            self.sigma2 = self.compute_mle_sigma2()[0,0]

        kernel_Cov_mat = self.compute_kernel_cov(self.X, self.theta)
        kernel_Vec_mat = self.compute_kernel_vec(self.X, current_point, self.theta)
        inv_kernel_Cov = np.linalg.inv(kernel_Cov_mat)

        mean_current = self.compute_mean(current_point)
        var_current = self.compute_var(current_point)

        grads_current_util = []

        for i in range(num_mc):
            z = np.random.normal(0, 1, 1)
            current_util = mean_current + np.sqrt(var_current)*z - y_max

            if current_util < zeroCheck: # pointwise utility function \hat{l}(x)<0 ===> grad = 0
                grad = np.zeros(shape=(self.dim, 1))
            else:
                grad_mean = self.compute_grad_mean(current_point)

                grad_var_part1 = self.sigma2 / np.sqrt(var_current)
                grad_var_part2 = self.compute_grad_kernel_vec(self.X, current_point, self.theta) 
                grad_var_part3 = np.matmul(inv_kernel_Cov, kernel_Vec_mat)
                grad_var = -1 * grad_var_part1 * np.matmul(grad_var_part2, grad_var_part3)

                grad = grad_mean + grad_var * z
            
            grads_current_util.append(grad)
        
        grad_current_util = np.mean(grads_current_util, axis=0) # take element-wise mean 

        return grad_current_util

    def find_NextBest_point(self):
        "find maximum point based on auxillary function"
        pass

    def plot(self, num_points=100, exp_ratio=1, confidence=0.9, kessis=[0.9]):
        "plot the acquisition function as well as ZeroGP in a figure with two figs"
        min_point = min(self.X)[0]
        max_point = max(self.X)[0]
        delta = max_point - min_point

        x_draw = np.linspace(min_point-exp_ratio*delta, max_point+exp_ratio*delta, num_points)

        # subplot 1: GProcess mean & confidence band
        y_mean = [self.compute_mean([ele]) for ele in x_draw]
        y_conf_int = [self.conf_interval([ele], confidence) for ele in x_draw]
        y_lower = [ele[0] for ele in y_conf_int]
        y_upper = [ele[1] for ele in y_conf_int]

        # subplot 2: EI AC function with multiple parameters 
        ac_values_lst = []
        if isinstance(kessis, list):
            for kessi in kessis:
                ac_kessi = [self.aux_func([ele], kessi) for ele in x_draw]
                ac_values_lst.append(ac_kessi)
        elif isinstance(kessis, float):
            ac_kessi = [self.aux_func([ele], kessis) for ele in x_draw]
            ac_values_lst.append(ac_kessi)
            kessis = [kessis]

        fig, (ax_gp, ax_ac) = plt.subplots(2, 1, sharex=True)

        ax_gp.set_title("GProcess Confidence Band")
        ax_gp.plot(x_draw, y_mean)
        ax_gp.fill_between(x_draw, y_lower, y_upper, alpha=0.2)
        ax_gp.plot(self.X, self.Y, 'o', color="tab:red")

        ax_ac.set_title("EI Acquisition Function")
        for ac_value, kessi in zip(ac_values_lst, kessis):
            ax_ac.plot(x_draw, ac_value, label="kessi: "+str(kessi))

        ax_ac.legend()
        fig.tight_layout()
        fig.savefig("./example_ac_ei.png")

        return 0


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

    UCB.plot(gammas=[0.8, 0.9, 1])

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

    EI.plot(kessis=[0.0, 0.1, 0.2])
