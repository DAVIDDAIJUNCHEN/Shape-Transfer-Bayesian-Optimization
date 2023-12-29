#!/usr/bin/env python3 
import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS


def write_exp_result(file, response, exp_point):
    "write experiemnt results to file"
    with open(file, 'a', encoding="utf-8") as fout:
        exp_line = str(response) + '\t' + '\t'.join([str(ele) for ele in exp_point]) + '\n'
        fout.writelines(exp_line)
    
    return 0

def get_best_point(file, response_col=0):
    "get the points with largest response"
    results = []
    with open(file, 'r', encoding="utf-8") as fin:
        for line in fin:
            if '#' not in line:
                line_split = line.split()
                point = line_split[:response_col] + line_split[(response_col+1):]
                results.append([(float(line_split[response_col]), point)])

    best = max(results)[0]
    best_response = float(best[0])
    best_point = [float(ele) for ele in best[1]]

    return best_point

def dist(pnt_a, pnt_b, l_n=2):
    "return l_n distance between pnt_a and pnt_b"
    unnormal_dist = np.sum([(a_i - b_i)**l_n for a_i, b_i in zip(pnt_a, pnt_b)])
    dist = unnormal_dist**(1/l_n)

    return dist

def draw_2d_lhd(file_sampling):
    "draw 2D LHD plot from sampling file"
    lst_v = []
    lst_x = []
    lst_y = []

    with open(file_sampling, "r", encoding="utf-8") as fin:
        for line in fin:
            if '#' not in line:
                lst_line = line.split()
                lst_v.append(lst_line[0])
                lst_x.append(lst_line[1])
                lst_y.append(lst_line[2])
            else:
                continue
    lst_x = [round(float(ele), 2) for ele in lst_x]
    lst_y = [round(float(ele), 2) for ele in lst_y]
    lst_v = [round(float(ele), 2) for ele in lst_v]

    fig, ax = plt.subplots()
    ax.scatter(np.array(lst_x), np.array(lst_y), vmin=-5, vmax=15)

    ax.set_title('LHD')
    
    ax.grid(True)
    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    file_dir = "./data/2D_Triple2Triple_10sample_2bad_prior_scaleTheta_maxminLHS/4"
    file_name = "simTriple2Triple2D_points_task0_sample.tsv"
    draw_2d_lhd(file_sampling=file_dir+'/'+file_name)

    # Maxmin LHD illustration    
    xlimits = np.array([[1.0, 4.0], [2.0, 3.0]])
    sampling = LHS(xlimits=xlimits, criterion='maximin')

    num = 10
    x = sampling(num)

    print(x.shape)

    plt.plot(x[:, 0], x[:, 1], "o")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
