#!/usr/bin/env python3 
import numpy as np

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

