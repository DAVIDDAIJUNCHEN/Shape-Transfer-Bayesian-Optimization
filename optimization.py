#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from operator import itemgetter
from gp import ZeroGProcess


class UpperConfidenceBound(ZeroGProcess):
    """
    class UpperConfidenceBound: 1. construct UCB auxillary function
                                2. find maximum point of UCB acquisition function
                                3. plot acquisition function & confidence bands
    """
    def __init__(self):
        super(UpperConfidenceBound, self).__init__()  # can use data from parent class

    def aux_func_ucb(self, current_point, current_gamma=0.9):
        """
        Acquisition function: a(x|D_t) = mean(x) + current_gamma * s2(x)
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

    def auto_grad_ucb(self, current_point):
        "compute gradient at current_point"

        pass 

    def find_NextBest_point_ucb(self):
        "find maximum point based on Acquisition function"
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
                ac_gamma = [self.aux_func_ucb([ele], gamma) for ele in x_draw]
                ac_values_lst.append(ac_gamma)
        elif isinstance(gammas, float):
            ac_gamma = [self.aux_func_ucb([ele], gammas) for ele in x_draw]
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
    class ExpectedImprovement: 1. construct EI Acquisition function 
                               2. find maximum point of EI Acquisition function
    """
    def __init__(self):  
        super(ExpectedImprovement, self).__init__()    # can use data from parent class 

    def aux_func_ei(self, current_point, kessi=0.0, zeroCheck=1e-13):
        """
        Acquisition function: a(x|D_t) = (mean(x) - y_max + kessi)*F(Z) + sqrt(sigma^2*s2(x))*f(Z)
        Z = (mean(x) - y_max + kessi) / sqrt(sigma^2*s2(x))
        """

        var_current = self.compute_var(current_point)

        if var_current < zeroCheck:
            aux_ei_current = 0.0
        else:
            y_max = max(self.Y)
            mean_current = self.compute_mean(current_point)

            Z_denom = mean_current - y_max + kessi
            Z = Z_denom / np.sqrt(var_current)

            aux_ei_current = Z_denom*norm.cdf(Z) + np.sqrt(var_current)*norm.pdf(Z)

        return aux_ei_current

    def auto_grad_ei(self, current_point, num_mc=1000, zeroCheck=1e-13):
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

    def find_best_from_point_ei(self, init_point, num_step=1000, thres=1e-3, learn_rate=0.1, beta_1=0.9, beta_2=0.999, epslon=1e-8):
        "find maximum point of Acquisition function from init_point by using ADAM algorithm"
        dim = self.dim
        assert(len(init_point) == dim)
        
        # initialize momentum & rmsp vector & gradient at init point
        m_t = np.zeros(shape=(dim, 1))
        gamma_t = np.zeros(shape=(dim, 1))
        point_current_vec = np.reshape(np.array(init_point), newshape=(dim, 1))
        point_current = init_point

        for t in range(num_step):
            grad_current = self.auto_grad_ei(point_current)
            m_t = beta_1*m_t + (1 - beta_1)*grad_current
            gamma_t = beta_2*gamma_t + (1 - beta_2)*grad_current**2

            # BC (bias correct) m_t and gamma_t
            m_t_BC = m_t / (1 - beta_1)
            gamma_t_BC = gamma_t / (1 - beta_2)  

            point_current_vec = point_current_vec + m_t_BC * learn_rate / np.sqrt(gamma_t_BC + epslon)
            point_current = np.reshape(point_current_vec, newshape=(1, dim)).tolist()
            point_current = point_current[0]
            aux_current = self.aux_func_ei(point_current)

        return point_current, aux_current

    def find_best_NextPoint_ei(self, init_points=None, num_step=1000, thres=1e-3, learn_rate=0.1, beta_1=0.9, beta_2=0.999, epslon=1e-8):
        """ find best next point of Acquisition function by starting from multi-points
            init_points: init_points = experiment points if None,
        """
        if init_points == None:
            init_points = self.X 
        
        best_points_aux = []

        for point_k in init_points:
            best_point_k, best_aux_k = self.find_best_from_point_ei(point_k, num_step, thres, learn_rate, beta_1, beta_2, epslon)
            best_points_aux.append((best_point_k, best_aux_k))

        best_point, best_aux = max(best_points_aux, key=itemgetter(1))
        print(best_points_aux)

        return best_point, best_aux

    def plot_ei(self, num_points=100, exp_ratio=1, confidence=0.9, kessis=[0.0], highlight_point=None):
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
                ac_kessi = [self.aux_func_ei([ele], kessi) for ele in x_draw]
                ac_values_lst.append(ac_kessi)
        elif isinstance(kessis, float):
            ac_kessi = [self.aux_func_ei([ele], kessis) for ele in x_draw]
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
        
        if highlight_point != None:
            ax_ac.plot(highlight_point[0], highlight_point[1], 'o', color="tab:red")

        ax_ac.legend()
        fig.tight_layout()
        fig.savefig("./example_ac_ei.png")

        return 0


class ShapeTransferBO(ExpectedImprovement, UpperConfidenceBound):
    """
    class ShapeTransferBO:
    
    """
    def __init__(self):
        super(ShapeTransferBO, self).__init__()
        self.zeroGP1 = None
        self.diffGP = None

    def build_task1_gp(self, file_exp_task1):
        "build ZeroGProcess for task1 with known experiment points"
        zeroGP1 = ZeroGProcess()
        zeroGP1.get_data_from_file(file_exp_task1)
        assert(len(zeroGP1.X) == len(zeroGP1.Y))

        self.zeroGP1 = zeroGP1

        return 0

    def build_diff_gp(self):
        "build ZeroGProcess on difference between task2 and task1_gp"

        diffGP = ZeroGProcess()
        assert(len(self.X) == len(self.Y))

        # compute difference between task2 and task1_gp
        X_task2 = self.X
        Y_task2 = self.Y

        diff_Y = []

        for point, y_task2 in zip(X_task2, Y_task2):
            mean_GP1_point = self.zeroGP1.compute_mean(point)
            diff_y_point = y_task2 - mean_GP1_point
            diff_Y.append(diff_y_point)
        
        diffGP.Y = diff_Y
        diffGP.X = self.X
        self.diffGP = diffGP

        return 0

    def compute_var(self, current_point, zeroCheck=1e-13):
        "compute the variance value at current_point"
        var_GP1 = self.zeroGP1.compute_var(current_point, zeroCheck)
        var_diffGP = self.diffGP.compute_var(current_point, zeroCheck)
        var_current = var_GP1 + var_diffGP

        if var_current < zeroCheck:
            return 0.0

        return var_current

    def compute_mean(self, current_point):
        "compute the mean value of GP1 + diffGP at current_point"
        mean_GP1 = self.zeroGP1.compute_mean(current_point)
        mean_diffGP = self.diffGP.compute_mean(current_point)

        mean_current = mean_GP1 + mean_diffGP

        return mean_current



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
    print("UCB({:.2f}) = {:.2f}".format(x1[0], UCB.aux_func_ucb(x1, gamma)))
    x2 = [10.4]
    print("UCB({:.2f}) = {:.2f}".format(x2[0], UCB.aux_func_ucb(x2, gamma)))

    #UCB.plot(gammas=[0.8, 0.9, 1])

    # Test STBO 
    STBO = ShapeTransferBO()
    STBO.get_data_from_file("./data/experiment_points_task2.tsv")
    STBO.build_task1_gp("./data/experiment_points_task1.tsv")
    STBO.build_diff_gp()

    print(STBO.X)
    print(STBO.Y)

    print(STBO.zeroGP1.X)
    print(STBO.zeroGP1.Y)

    print(STBO.diffGP.X)
    print(STBO.diffGP.Y)

    x1 = [1.5]
    print(STBO.aux_func_ei(x1))
    STBO.plot_ei(highlight_point=[5, STBO.aux_func_ei([5])])
