#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from analyze_results import collect_file, run_statistics
from optimization import ExpectedImprovement, ShapeTransferBO
from gp import ZeroGProcess
from simfun import two_exp_mu, tri_exp_mu


def show_medium_percentile_errorbar(dct_medium_perc1, dct_medium_perc2, title, fig_name, STBO="scratch"):
    "plot lines with error bar based on medium and percentile"

    #fig = plt.figure(figsize=plt.figaspect(0.3))
    #fig.suptitle(title[0])

    dct_medium_perc = [dct_medium_perc1, dct_medium_perc2]

    fig, axs = plt.subplots(1, 2, figsize=(10, 2))

    for i in range(2):
        axs[i].set_title(title[i+1], fontsize=17)
        if i == 1:
          fig.legend(loc=9, bbox_to_anchor=(0.25, 0.5, 0.5, 0.5), ncol=8, fontsize='x-large')

        for item in sorted(dct_medium_perc[i].items()):
            if "1_" in item[0]:
                continue 
            elif "bcbo" in item[0]:
                continue
            elif "MHGP" in item[0]:
                continue
            elif "gp_from" in item[0]:
                continue
            elif "STBO" in item[0] and STBO == "scratch":
                continue
            elif "stbo" in item[0] and STBO == "package":
                continue

            x_draw = np.arange(len(item[1]))
            x_draw = [ele + 1 for ele in x_draw]
            y_medium = [ele[1] for ele in item[1]]
            y_perc25 = [ele[1] - ele[0] for ele in item[1]]
            y_perc75 = [ele[2] - ele[1] for ele in item[1]]
            asymmetric_error = [y_perc25, y_perc75]
            
            if "task2_DiffGP_from_" in item[0] and "0_" in item[0]:
                label = "Diff-GP"
                fmt = '-.^'
                color = "blue"             
            elif "task2_BHGP_from_" in item[0] and "0_" in item[0]:
                label = "BHGP"
                fmt = '-o'
                color = "brown"
            elif "task2_MHGP_from_" in item[0] and "0_" in item[0]:
                label = "MHGP"
                fmt = '--s'
                color = "black"
            elif "task2_HGP_from_" in item[0] and "0_" in item[0]:
                label = "HGP"
                fmt = '-.^'
                color = "green"
            elif "task2_MTGP_from_" in item[0] and "0_" in item[0]:
                label = "MTGP"
                fmt = '-o'
                color = "grey"
            elif "task2_SHGP_from_" in item[0] and "0_" in item[0]:
                label = "SHGP"
                fmt = '--s'
                color = "orange"
            elif "task2_stbo_from_" in item[0] and "0_" in item[0] and STBO == "scratch":
                label = "STBO"
                fmt = '--s'
                color = "red"             
            elif "task2_STBO_from_" in item[0] and "0_" in item[0] and STBO == "package":
                label = "STBO"
                fmt = '--s'
                color = "red"      
            elif "task2_WSGP_from_" in item[0] and "0_" in item[0]:
                label = "WSGP"
                fmt = "-s"
                color = "violet"

            axs[i].errorbar(x_draw, y_medium, yerr=asymmetric_error, label=label, fmt=fmt, color=color)
            axs[i].set_xticks(np.arange(0, len(x_draw)+1, 5))
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


if __name__ == "__main__":
    # simulation 1: 1D sampling triple2triple
    in_dir_d2d = "./data/sampling_experiments/Dimension-1"
    out_dir_d2d = "./simulation_results/sampling_experiments/Dimension-1"

    file_lsts_dissimilar = collect_file(in_dir_d2d, topic="from_rand", similar="ge1") # Tao >= 1
    file_lsts_similar    = collect_file(in_dir_d2d, topic="from_rand", similar="lt1")     # Tao <  1

    _, dct_medium_perc_dissimilar = run_statistics(file_lsts_dissimilar, out_dir_d2d)
    _, dct_medium_perc_similar = run_statistics(file_lsts_similar, out_dir_d2d)

    fig_name_medium = "./images/sampling_simulation_1D_rand_paper.pdf"

    title = ["Simulation 1: target function sampled from triple modules", "Similar", "Dissimilar"]
    show_medium_percentile_errorbar(dct_medium_perc_similar, dct_medium_perc_dissimilar, title, fig_name=fig_name_medium, STBO="scratch")

    # simulation 2: 2D sampling triple2triple
    # part 1: 2D sampling from gp
    in_dir_d2d = "./data/sampling_experiments/Dimension-2"
    out_dir_d2d = "./simulation_results/sampling_experiments/Dimension-2"

    file_lsts_dissimilar = collect_file(in_dir_d2d, topic="from_gp", similar="ge1") # Tao >= 1
    file_lsts_similar    = collect_file(in_dir_d2d, topic="from_gp", similar="lt1")     # Tao <  1

    _, dct_medium_perc_dissimilar = run_statistics(file_lsts_dissimilar, out_dir_d2d)
    _, dct_medium_perc_similar = run_statistics(file_lsts_similar, out_dir_d2d)

    fig_name_medium = "./images/sampling_simulation_2D_gp_paper.pdf"

    title = ["Simulation 1: target function sampled from triple modules", "Similar", "Dissimilar"]
    show_medium_percentile_errorbar(dct_medium_perc_similar, dct_medium_perc_dissimilar, title, fig_name=fig_name_medium, STBO="package")

    # part 2: 2D sampling from random
    in_dir_d2d = "./data/sampling_experiments/Dimension-2"
    out_dir_d2d = "./simulation_results/sampling_experiments/Dimension-2"

    file_lsts_dissimilar = collect_file(in_dir_d2d, topic="from_rand", similar="ge1") # Tao >= 1
    file_lsts_similar    = collect_file(in_dir_d2d, topic="from_rand", similar="lt1")     # Tao <  1

    _, dct_medium_perc_dissimilar = run_statistics(file_lsts_dissimilar, out_dir_d2d)
    _, dct_medium_perc_similar = run_statistics(file_lsts_similar, out_dir_d2d)

    fig_name_medium = "./images/sampling_simulation_2D_rand_paper.pdf"

    title = ["Simulation 1: target function sampled from triple modules", "Similar", "Dissimilar"]
    show_medium_percentile_errorbar(dct_medium_perc_similar, dct_medium_perc_dissimilar, title, fig_name=fig_name_medium, STBO="scratch")

