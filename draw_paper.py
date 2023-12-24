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
                label = "Diff-GP from rand"
                fmt = '-.^'
                color = "blue"
            elif "task2_stbo_from_rand.tsv" in item[0]:
                label = "STBO from rand"
                fmt = '-o'
                color = "green"
            elif "task2_gp_from_rand.tsv" in item[0]:
                label = "EI from rand"
                fmt = '--s'
                color = "red"
            elif "task2_bcbo_from_gp.tsv" in item[0]:
                label = "Diff-GP from EI"
                fmt = '-.^'
                color = "blue"
            elif "task2_stbo_from_gp.tsv" in item[0]:
                label = "STBO from EI"
                fmt = '-o'
                color = "green"
            elif "task2_gp_from_gp.tsv" in item[0]:
                label = "EI from EI"
                fmt = '--s'
                color = "red"
            elif "task2_gp_from_cold.tsv" in item[0]:
                label = "EI from cold"
                fmt = '--s'
                color = "red"

            ax.errorbar(x_draw, y_medium, yerr=asymmetric_error, label=label, fmt=fmt, color=color)
            ax.set_xticks(np.arange(0, 21, 5))
        plt.legend(loc=4)

    plt.gcf().set_size_inches(20, 5)
    plt.show()
    plt.savefig(fig_name)
    
    return 0


# Part 3: EXP function family
def show_EXP_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name):
    "plot lines with error bar based on medium and percentile"

    fig = plt.figure(figsize=plt.figaspect(0.7))

    dct_medium_perc = [dct_medium_perc1, dct_medium_perc2, dct_medium_perc3]

    subfig1, subfig2 = fig.subfigures(2, 1)

    subfigs = [subfig1, subfig2]

    for row in range(2):
        subfigs[row].suptitle(title[row])
        for col in range(3):
            ax = subfigs[row].add_subplot(1, 3, col + 1)
            ax.set_title("$\mu=$"+title[2 + col])

            for item in sorted(dct_medium_perc[col][row].items()):
                x_draw = np.arange(len(item[1]))
                x_draw = [ele + 1 for ele in x_draw]
                y_medium = [ele[1] for ele in item[1]]
                y_perc25 = [ele[1] - ele[0] for ele in item[1]]
                y_perc75 = [ele[2] - ele[1] for ele in item[1]]
                asymmetric_error = [y_perc25, y_perc75]

                if "task2_bcbo_from_rand.tsv" in item[0]:
                    label = "Diff-GP"
                    fmt = '-.^'
                    color = "blue"
                elif "task2_stbo_from_rand.tsv" in item[0]:
                    label = "STBO"
                    fmt = '-o'
                    color = "green"
                elif "task2_gp_from_rand.tsv" in item[0]:
                    label = "EI from rand"
                    fmt = '--s'
                    color = "red"
                elif "task2_gp_from_cold.tsv" in item[0]:
                    label = "EI from cold"
                    fmt = '--s'
                    color = "red"

                ax.errorbar(x_draw, y_medium, yerr=asymmetric_error, label=label, fmt=fmt, color=color)
                ax.set_xticks(np.arange(0, 21, 5))

            plt.legend(loc=4)

    plt.gcf().set_size_inches(20, 11)
    plt.show()
    plt.savefig(fig_name)
    
    return 0


if __name__ == "__main__":
    # Exponential family
    size = 4
    mu1 = [0.1,0.1]; mu2 = [0.4163, 0.4163]; mu3 = [1.0, 1.0]
    theta = 0.5
    show_exp(mu1, mu2, mu3, theta, x_nums=200, y_nums=200, x_low=-size,x_up=size,y_low=-size, y_up=size)

    # simulation 1: Double2Double
    in_dir1 = "./data/Double2Double"
    out_dir1 = "./simulation_results/Double2Double"

    file_lsts_stbo1 = collect_file(in_dir1, "stbo_from_rand")
    file_lsts_cold1 = collect_file(in_dir1, "from_cold")
    file_lsts_stbo1.extend(file_lsts_cold1)
    file_lsts_1 = file_lsts_stbo1

    _, dct_medium_perc1 = run_statistics(file_lsts_1, out_dir1)

    file_lsts_2 = collect_file(in_dir1, topic="from_rand")
    file_lsts_3 = collect_file(in_dir1, topic="from_gp")

    _, dct_medium_perc2 = run_statistics(file_lsts_2, out_dir1)
    _, dct_medium_perc3 = run_statistics(file_lsts_3, out_dir1)

    fig_name_medium = "./images/double2double_paper.pdf"

    title = ["Simulation 1: from double modals to double modals", "transfer vs non-transfer", "start from rand", "start from gp"]
    show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name=fig_name_medium)

    # simulation 2: Triple2Double 
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

    fig_name_medium = "./images/triple2double_paper.pdf"

    title = ["Simulation 2: from triple modals to double modals", "transfer vs non-transfer", "start from rand", "start from gp"]
    show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name=fig_name_medium)

    # simulation 2: Double2Triple
    in_dir1 = "./data/Double2Triple_0.5"
    out_dir1 = "./simulation_results/Double2Triple_0.5"

    file_lsts_stbo1 = collect_file(in_dir1, "stbo_from_rand")
    file_lsts_cold1 = collect_file(in_dir1, "from_cold")
    file_lsts_stbo1.extend(file_lsts_cold1)
    file_lsts_1 = file_lsts_stbo1

    _, dct_medium_perc1 = run_statistics(file_lsts_1, out_dir1)

    file_lsts_2 = collect_file(in_dir1, topic="from_rand")
    file_lsts_3 = collect_file(in_dir1, topic="from_gp")

    _, dct_medium_perc2 = run_statistics(file_lsts_2, out_dir1)
    _, dct_medium_perc3 = run_statistics(file_lsts_3, out_dir1)

    fig_name_medium = "./images/double2triple_paper.pdf"

    title = ["Simulation 2: from double modals to triple modals", "transfer vs non-transfer", "start from rand", "start from gp"]
    show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name=fig_name_medium)


    # simulation 3: 2D Triple2Triple
    in_dir1 = "./data/2D_Triple2Triple"
    out_dir1 = "./simulation_results/2D_Triple2Triple"

    file_lsts_stbo1 = collect_file(in_dir1, "stbo_from_rand")
    file_lsts_cold1 = collect_file(in_dir1, "from_cold")
    file_lsts_stbo1.extend(file_lsts_cold1)
    file_lsts_1 = file_lsts_stbo1

    _, dct_medium_perc1 = run_statistics(file_lsts_1, out_dir1)

    file_lsts_2 = collect_file(in_dir1, topic="from_rand")
    file_lsts_3 = collect_file(in_dir1, topic="from_gp")

    _, dct_medium_perc2 = run_statistics(file_lsts_2, out_dir1)
    _, dct_medium_perc3 = run_statistics(file_lsts_3, out_dir1)

    fig_name_medium = "./images/2d_triple2triple_paper.pdf"

    title = ["Simulation 3: from 2d triple modals to triple modals", "transfer vs non-transfer", "start from rand", "start from gp"]
    show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name=fig_name_medium)


    # simulation 4: EXP 
    thetas = [0.87, 1, 1.414]
    means = [(0.435, 0.74, 1.74), (0.5, 0.832555, 2), (0.707, 1.177, 2.828)]
    #thetas = [0.5, 1, 1.414]
    #means = [(0.1, 0.4163, 1.0), (0.5, 0.832555, 2), (0.707, 1.177, 2.828)]

    for i in range(3):
        theta = thetas[i]
        mean_tuple = means[i]
        # mean 1
        in_dir1 = "./data/EXP_mu2_" + str(mean_tuple[0]) + "_" + str(mean_tuple[0]) + "_theta_" + str(theta)
        out_dir1 = "./simulation_results/EXP_theta_" + str(theta)

        file_lsts_stbo1 = collect_file(in_dir1, "stbo_from_rand")
        file_lsts_cold1 = collect_file(in_dir1, "from_cold")
        file_lsts_stbo1.extend(file_lsts_cold1)

        file_lsts_1_1 = file_lsts_stbo1
        _, dct_medium_perc1_1 = run_statistics(file_lsts_1_1, out_dir1)        

        file_lsts_1_2 = collect_file(in_dir1, topic="from_rand")
        _, dct_medium_perc1_2 = run_statistics(file_lsts_1_2, out_dir1)

        dct_medium_perc1 = [dct_medium_perc1_1, dct_medium_perc1_2]

        # mean 2
        in_dir2 = "./data/EXP_mu2_" + str(mean_tuple[1]) + "_" + str(mean_tuple[1]) + "_theta_" + str(theta)
        out_dir2 = "./simulation_results/EXP_theta_" + str(theta)

        file_lsts_stbo2 = collect_file(in_dir2, "stbo_from_rand")
        file_lsts_cold2 = collect_file(in_dir2, "from_cold")
        file_lsts_stbo2.extend(file_lsts_cold2)

        file_lsts_2_1 = file_lsts_stbo2
        _, dct_medium_perc2_1 = run_statistics(file_lsts_2_1, out_dir2)        

        file_lsts_2_2 = collect_file(in_dir2, topic="from_rand")
        _, dct_medium_perc2_2 = run_statistics(file_lsts_2_2, out_dir2)     

        dct_medium_perc2 = [dct_medium_perc2_1, dct_medium_perc2_2]

        # mean 3
        in_dir3 = "./data/EXP_mu2_" + str(mean_tuple[2]) + "_" + str(mean_tuple[2]) + "_theta_" + str(theta)
        out_dir3 = "./simulation_results/EXP_theta_" + str(theta)

        file_lsts_stbo3 = collect_file(in_dir3, "stbo_from_rand")
        file_lsts_cold3 = collect_file(in_dir3, "from_cold")
        file_lsts_stbo3.extend(file_lsts_cold3)

        file_lsts_3_1 = file_lsts_stbo3
        _, dct_medium_perc3_1 = run_statistics(file_lsts_3_1, out_dir3)        

        file_lsts_3_2 = collect_file(in_dir3, topic="from_rand")
        _, dct_medium_perc3_2 = run_statistics(file_lsts_3_2, out_dir3)  

        dct_medium_perc3 = [dct_medium_perc3_1, dct_medium_perc3_2]

        fig_name_medium = "./images/simEXP_theta_" + str(theta) + ".pdf"

        t0 = "(" + str(mean_tuple[0]) + ", " + str(mean_tuple[0]) + ")"
        t1 = "(" + str(mean_tuple[1]) + ", " + str(mean_tuple[1]) + ")"
        t2 = "(" + str(mean_tuple[2]) + ", " + str(mean_tuple[2]) + ")"
        title = ["transfer vs. non-transfer methods", "transfer methods", t0, t1, t2]

        show_EXP_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name=fig_name_medium)
