#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from analyze_results import collect_file, run_statistics
from optimization import ExpectedImprovement, ShapeTransferBO
from gp import ZeroGProcess
from simfun import two_exp_mu, tri_exp_mu


def show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, dct_medium_perc3, title, fig_name):
    "plot lines with error bar based on medium and percentile"

    #fig = plt.figure(figsize=plt.figaspect(0.3))
    #fig.suptitle(title[0])

    dct_medium_perc = [dct_medium_perc1, dct_medium_perc2, dct_medium_perc3]

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    for i in range(3):
        axs[i].set_title(title[i+1], fontsize=20)
        if i == 1:
          fig.legend(loc=9, bbox_to_anchor=(0.25, 0.5, 0.5, 0.5), ncol=7, fontsize='large')

        for item in sorted(dct_medium_perc[i].items()):
            x_draw = np.arange(len(item[1]))
            x_draw = [ele + 1 for ele in x_draw]
            y_medium = [ele[1] for ele in item[1]]
            y_perc25 = [ele[1] - ele[0] for ele in item[1]]
            y_perc75 = [ele[2] - ele[1] for ele in item[1]]
            asymmetric_error = [y_perc25, y_perc75]
            
            if "task2_bcbo_from_" in item[0]:
                label = "Diff-GP"
                fmt = '-.^'
                color = "blue"
            elif "task2_BHGP_from_" in item[0]:
                label = "BHGP"
                fmt = '-o'
                color = "brown"
            elif "task2_gp_from_" in item[0]:
                label = "EI"
                fmt = '--s'
                color = "black"
                continue
            elif "task2_HGP_from_" in item[0]:
                label = "HGP"
                fmt = '-.^'
                color = "green"
            elif "task2_MTGP_from_" in item[0]:
                label = "MTGP"
                fmt = '-o'
                color = "grey"
            elif "task2_SHGP_from_" in item[0]:
                label = "SHGP"
                fmt = '--s'
                color = "orange"
            elif "task2_stbo_from_" in item[0]:
                label = "STBO"
                fmt = '--s'
                color = "red"
            elif "task2_WSGP_from_" in item[0]:
                label = "WSGP"
                fmt = "-s"
                color = "violet"

            axs[i].errorbar(x_draw, y_medium, yerr=asymmetric_error, label=label, fmt=fmt, color=color)
            axs[i].set_xticks(np.arange(0, 21, 5))
            axs[i].tick_params(axis='x', labelsize=18)
            axs[i].tick_params(axis='y', labelsize=18)
            axs[i].set_xlabel('Steps', fontsize=20)
            if i == 0:
                axs[i].set_ylabel('Function value', fontsize=20)
        #plt.legend(loc=4)
    
    plt.gcf().set_size_inches(25, 6)

    plt.show()
    plt.savefig(fig_name)
    
    return 0


def show_medium_regret(dct_medium_perc, title, fig_name):
    "plot lines with error bar based on medium and regret"

    dct_medium_perc = [dct_medium_perc]

    fig, axs = plt.subplots(1, 2, figsize=(10, 2))

    # Left subplot: 10 - RMSE_CV
    axs[0].set_title(title[1])
    max_cv_california = 0

    for item in sorted(dct_medium_perc[0].items()):
        x_draw = np.arange(len(item[1]))
        x_draw = [ele + 1 for ele in x_draw]
        y_medium = [ele[1] for ele in item[1]]
        y_perc25 = [ele[1] - ele[0] for ele in item[1]]
        y_perc75 = [ele[2] - ele[1] for ele in item[1]]
        asymmetric_error = [y_perc25, y_perc75]
        if max_cv_california < max(y_medium):
            max_cv_california = max(y_medium)

        if "task2_bcbo_from_" in item[0]:
            label = "Diff-GP"
            fmt = '-.^'
            color = "blue"
        elif "task2_BHGP_from_" in item[0]:
            label = "BHGP"
            fmt = '-o'
            color = "brown"
        # elif "task2_gp_from_" in item[0]:
        #     label = "EI"
        #     fmt = '--s'
        #     color = "black"
        elif "task2_HGP_from_" in item[0]:
            label = "HGP"
            fmt = '-.^'
            color = "green"
        elif "task2_MTGP_from_" in item[0]:
            label = "MTGP"
            fmt = '-o'
            color = "grey"
        elif "task2_SHGP_from_" in item[0]:
            label = "SHGP"
            fmt = '--s'
            color = "orange"
        elif "task2_stbo_from_" in item[0]:
            label = "STBO"
            fmt = '--s'
            color = "red"
        elif "task2_WSGP_from_" in item[0]:
            label = "WSGP"
            fmt = "-s"
            color = "violet"

        axs[0].errorbar(x_draw, y_medium, yerr=asymmetric_error, label=label, fmt=fmt, color=color)
        axs[0].set_xticks(np.arange(0, 11, 1))
        axs[0].set_xlabel('Steps')

        axs[0].set_ylabel('10 - RMSE')

    fig.legend(loc=9, bbox_to_anchor=(0.25, 0.5, 0.5, 0.5), ncol=8, fontsize='large')
    # Right subplot: simple regret
    axs[1].set_title(title[2])

    for item in sorted(dct_medium_perc[0].items()):
        x_draw = np.arange(len(item[1]))
        x_draw = [ele + 1 for ele in x_draw]
        y_medium = [ele[1] for ele in item[1]]
        y_perc25 = [ele[1] - ele[0] for ele in item[1]]
        y_perc75 = [ele[2] - ele[1] for ele in item[1]]
        asymmetric_error = [y_perc25, y_perc75]
        regret_medium = [max_cv_california - cv for cv in y_medium]

        if "task2_bcbo_from_" in item[0]:
            label = "Diff-GP"
            fmt = '-.^'
            color = "blue"
        elif "task2_BHGP_from_" in item[0]:
            label = "BHGP"
            fmt = '-o'
            color = "brown"
        # elif "task2_gp_from_" in item[0]:
        #     label = "EI"
        #     fmt = '--s'
        #     color = "black"
        elif "task2_HGP_from_" in item[0]:
            label = "HGP"
            fmt = '-.^'
            color = "green"
        elif "task2_MTGP_from_" in item[0]:
            label = "MTGP"
            fmt = '-o'
            color = "grey"
        elif "task2_SHGP_from_" in item[0]:
            label = "SHGP"
            fmt = '--s'
            color = "orange"
        elif "task2_stbo_from_" in item[0]:
            label = "STBO"
            fmt = '--s'
            color = "red"
        elif "task2_WSGP_from_" in item[0]:
            label = "WSGP"
            fmt = "-s"
            color = "violet"

        axs[1].errorbar(x_draw, regret_medium, yerr=asymmetric_error, label=label, fmt=fmt, color=color)
        axs[1].set_xticks(np.arange(0, 11, 1))
        axs[1].set_xlabel('Steps')

        axs[1].set_ylabel('Simple regret')

    plt.gcf().set_size_inches(15, 6)

    plt.show()
    plt.savefig(fig_name)
    
    return 0

if __name__ == "__main__":
    # simulation 1: Double2Double
    in_dir_d2d = "./data/Double2Double"
    out_dir_d2d = "./simulation_results/Double2Double"

    in_dir_d2t = "./data/Double2Triple"
    out_dir_d2t = "./simulation_results/Double2Triple"

    in_dir_t2t = "./data/2D_Triple2Triple"
    out_dir_t2t = "./data/2D_Triple2Triple"

    file_lsts_d2d = collect_file(in_dir_d2d, topic="from_gp")
    file_lsts_d2t = collect_file(in_dir_d2t, topic="from_gp")
    file_lsts_t2t = collect_file(in_dir_t2t, topic="from_gp")

    _, dct_medium_perc_d2d = run_statistics(file_lsts_d2d, out_dir_d2d)
    _, dct_medium_perc_d2t = run_statistics(file_lsts_d2t, out_dir_d2t)
    _, dct_medium_perc_t2t = run_statistics(file_lsts_t2t, out_dir_t2t)

    fig_name_medium = "./images/3types_simulation_paper.pdf"

    title = ["Simulation 1: from double modals to double modals", "Toy example 2", "Toy example 3", "Toy example 4"]
    show_medium_percentile_errorbar(dct_medium_perc_d2d, dct_medium_perc_d2t, dct_medium_perc_t2t, title, fig_name=fig_name_medium)

    # Simulation 2: Exponential family: theta = 0.87
    in_dir_close = "./data/EXP_mu2_0.435_0.435_theta_0.87"
    out_dir_close = "./simulation_results/EXP_mu2_0.435_0.435_theta_0.87"

    in_dir_mid = "./data/EXP_mu2_0.74_0.74_theta_0.87"
    out_dir_mid = "./simulation_results/EXP_mu2_0.74_0.74_theta_0.87"

    in_dir_far = "./data/EXP_mu2_1.74_1.74_theta_0.87"
    out_dir_far = "./simulation_results/EXP_mu2_1.74_1.74_theta_0.87"

    file_lsts_close = collect_file(in_dir_close, topic="from_rand")
    file_lsts_mid = collect_file(in_dir_mid, topic="from_rand")
    file_lsts_far = collect_file(in_dir_far, topic="from_rand")

    _, dct_medium_perc_close = run_statistics(file_lsts_close, out_dir_d2d)
    _, dct_medium_perc_mid = run_statistics(file_lsts_mid, out_dir_d2t)
    _, dct_medium_perc_far = run_statistics(file_lsts_far, out_dir_t2t)

    fig_name_medium = "./images/EXP_theta_0.87_paper.pdf"

    title = ["Simulation 1: from double modals to double modals", "$\mu=(0.435, 0.435)$", "$\mu=(0.74, 0.74)$", "$\mu=(1.74, 1.74)$"]
    show_medium_percentile_errorbar(dct_medium_perc_close, dct_medium_perc_mid, dct_medium_perc_far, title, fig_name=fig_name_medium)    


    # Simulation 3: Exponential family: theta = 1
    in_dir_close = "./data/EXP_mu2_0.5_0.5_theta_1"
    out_dir_close = "./simulation_results/EXP_mu2_0.5_0.5_theta_1"

    in_dir_mid = "./data/EXP_mu2_0.832555_0.832555_theta_1"
    out_dir_mid = "./simulation_results/EXP_mu2_0.832555_0.832555_theta_1"

    in_dir_far = "./data/EXP_mu2_2_2_theta_1"
    out_dir_far = "./simulation_results/EXP_mu2_2_2_theta_1"

    file_lsts_close = collect_file(in_dir_close, topic="from_rand")
    file_lsts_mid = collect_file(in_dir_mid, topic="from_rand")
    file_lsts_far = collect_file(in_dir_far, topic="from_rand")

    _, dct_medium_perc_close = run_statistics(file_lsts_close, out_dir_d2d)
    _, dct_medium_perc_mid = run_statistics(file_lsts_mid, out_dir_d2t)
    _, dct_medium_perc_far = run_statistics(file_lsts_far, out_dir_t2t)

    fig_name_medium = "./images/EXP_theta_1_paper.pdf"

    title = ["Simulation 1: from double modals to double modals", "$\mu=(0.5, 0.5)$", "$\mu=(0.8325, 0.8325)$", "$\mu=(2, 2)$"]
    show_medium_percentile_errorbar(dct_medium_perc_close, dct_medium_perc_mid, dct_medium_perc_far, title, fig_name=fig_name_medium)    


    # Simulation 4: Exponential family: theta = 1.414
    in_dir_close = "./data/EXP_mu2_0.707_0.707_theta_1.414"
    out_dir_close = "./simulation_results/EXP_mu2_0.707_0.707_theta_1.414"

    in_dir_mid = "./data/EXP_mu2_1.177_1.177_theta_1.414"
    out_dir_mid = "./simulation_results/EXP_mu2_1.177_1.177_theta_1.414"

    in_dir_far = "./data/EXP_mu2_2.828_2.828_theta_1.414"
    out_dir_far = "./simulation_results/EXP_mu2_2.828_2.828_theta_1.414"

    file_lsts_close = collect_file(in_dir_close, topic="from_rand")
    file_lsts_mid = collect_file(in_dir_mid, topic="from_rand")
    file_lsts_far = collect_file(in_dir_far, topic="from_rand")

    _, dct_medium_perc_close = run_statistics(file_lsts_close, out_dir_d2d)
    _, dct_medium_perc_mid = run_statistics(file_lsts_mid, out_dir_d2t)
    _, dct_medium_perc_far = run_statistics(file_lsts_far, out_dir_t2t)

    fig_name_medium = "./images/EXP_theta_1.414_paper.pdf"

    title = ["Simulation 1: from double modals to double modals", "$\mu=(0.707, 0.707)$", "$\mu=(1.177, 1.177)$", "$\mu=(2.828, 2.828)$"]
    show_medium_percentile_errorbar(dct_medium_perc_close, dct_medium_perc_mid, dct_medium_perc_far, title, fig_name=fig_name_medium)    

    # Simulation 5: Boston House price ==> California House price
    in_dir_xgb = "./data/Xgb_5d_20Source"
    out_dir_xgb = "./simulation_results/Xgb_5d_20Source"

    file_lsts_xgb = collect_file(in_dir_xgb, topic="from_gp")

    _, dct_medium_perc = run_statistics(file_lsts_xgb, out_dir_xgb)

    fig_name_medium = "./images/XGB_5d_20Source_paper.pdf"

    title = ["Simulation 1: from double modals to double modals", "(a)", "(b)"]
    show_medium_regret(dct_medium_perc, title, fig_name=fig_name_medium)    
    
