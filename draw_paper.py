#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from analyze_results import collect_file, run_statistics


# Part 1: images for simulation 1
def show_exp(mu1=[0.1, 0.1], mu2=[0.4163, 0.4163], m3=[1.0,1.0], theta=0.5, x_low=-1, x_up=1, y_low=-1, y_up=1, x_nums=100, y_nums=100):

    X = np.arange(x_low, x_up, 0.25)
    Y = np.arange(y_low, y_up, 0.25)
    X, Y = np.meshgrid(X, Y)

    exp0_norm2 = X**2 + Y**2 
    exp1_norm2 = (X- mu1[0])**2 + (Y - mu1[1])**2
    exp2_norm2 = (X- mu2[0])**2 + (Y - mu2[1])**2
    exp3_norm2 = (X- mu3[0])**2 + (Y - mu3[1])**2

    exp0 = np.exp(-0.5 * exp0_norm2 / theta**2)
    exp1 = np.exp(-0.5 * exp1_norm2 / theta**2)
    exp2 = np.exp(-0.5 * exp2_norm2 / theta**2)
    exp3 = np.exp(-0.5 * exp3_norm2 / theta**2)

    exp = [exp1, exp2, exp3]
    mu = [mu1, mu2, mu3]

    fig = plt.figure(figsize=plt.figaspect(0.5))

    for i in range(3):     
        Z1 = exp0
        Z2 = exp[i]
    
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_wireframe(X, Y, Z1, rstride=2, cstride=2, color="red")
        ax.plot_wireframe(X, Y, Z2, rstride=2, cstride=2)
    
        ax.set_title(r"$\mu$"+"=("+str(mu[i][0]) +","+str(mu[i][1])+")", fontsize=25)


    plt.show()

    fig.savefig("./images/exp_3d_paper")

    return 0

# Part 2: Double2Double
def show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name):
    "plot lines with error bar based on medium and percentile"

    fig = plt.figure(figsize=plt.figaspect(0.3))
    fig.suptitle(title[0])

    dct_medium_perc = [dct_medium_perc1, dct_medium_perc2, dct_medium_perc3]

    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.set_title(title[i+1])

        for item in sorted(dct_medium_perc[i].items()):
            x_draw = np.arange(len(item[1]))
            x_draw = [ele + 1 for ele in x_draw]
            y_medium = [ele[1] for ele in item[1]]
            y_perc25 = [ele[1] - ele[0] for ele in item[1]]
            y_perc75 = [ele[2] - ele[1] for ele in item[1]]
            asymmetric_error = [y_perc25, y_perc75]
            
            if "task2_bcbo_from_rand.tsv" in item[0]:
                label = "BCBO from rand"
            elif "task2_stbo_from_rand.tsv" in item[0]:
                label = "STBO from rand"
            elif "task2_gp_from_rand.tsv" in item[0]:
                label = "EI from rand"
            elif "task2_bcbo_from_gp.tsv" in item[0]:
                label = "BCBO from gp"
            elif "task2_stbo_from_gp.tsv" in item[0]:
                label = "STBO from gp"
            elif "task2_gp_from_gp.tsv" in item[0]:
                label = "EI from gp"
            elif "task2_gp_from_cold.tsv" in item[0]:
                label = "EI from cold"

            ax.errorbar(x_draw, y_medium, yerr=asymmetric_error, label=label, fmt='-o')

        plt.legend()
    
    plt.show()
    plt.savefig(fig_name)
    
    return 0


if __name__ == "__main__":
    # Exponential family
    size = 4
    mu1 = [0.1,0.1]; mu2 = [0.4163, 0.4163]; mu3 = [1.0, 1.0]
    theta = 0.5
    show_exp(mu1, mu2, mu3, theta, x_nums=200, y_nums=200, x_low=-size,x_up=size,y_low=-size, y_up=size)

    # Double2Double 
    in_dir1 = "./data/Triple2Double"
    out_dir1 = "./simulation_results/Triple2Double"

    file_lsts_stbo1 = collect_file(in_dir1, "stbo_from_rand")
    file_lsts_cold1 = collect_file(in_dir1, "from_cold")
    file_lsts_stbo1.extend(file_lsts_cold1)
    file_lsts_1 = file_lsts_stbo1

    _, dct_medium_perc1 = run_statistics(file_lsts_1, out_dir1)

    file_lsts_2 = collect_file(in_dir1, topic="from_rand")
    file_lsts_3 = collect_file(in_dir1, topic="from_gp")

    _, dct_medium_perc2 = run_statistics(file_lsts_2, out_dir1)
    _, dct_medium_perc3 = run_statistics(file_lsts_3, out_dir1)

    fig_name_medium = "./images/triple2double_paper.png"

    title = ["Simulation 3: from triple modals to double modals", "transfer vs non-transfer", "start from rand", "start from gp"]
    show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name=fig_name_medium)
