#!/usr/bin/env python3 
import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS


def find_max_min_of_each_component(lst, max=True):
    "return [7, 8, 9] for find_max_of_each_component([1,2,3],[7,8,9])"
    
    if max:
        max_values = []
        for i in range(len(lst[0])):
            max_val = float('-inf')
            for sub_lst in lst:
                if sub_lst[i] > max_val:
                    max_val = sub_lst[i]
            max_values.append(max_val)
        return max_values
    else:
        min_values = []
        for i in range(len(lst[0])):
            min_val = float("inf")
            for sub_lst in lst:
                if sub_lst[i] < min_val:
                    min_val = sub_lst[i]
            min_values.append(min_val)

        return min_values

def check_inBounds(pnt, l_bounds, u_bounds):
    "check if point pnt lies in zone with boundary (l_bounds, u_bounds)"
    assert(len(pnt)==len(l_bounds)) 
    assert(len(pnt)==len(u_bounds))
    in_zone = True

    for i in range(len(pnt)):
        if pnt[i]<l_bounds[i] or pnt[i]>u_bounds[i]:
            in_zone = False
            break

    return in_zone

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

def get_all_points(file, response_col=0):
    "get all the points"
    all_points = []
    with open(file, 'r', encoding="utf-8") as fin:
        for line in fin:
            if '#' not in line:
                line_split = line.split()
                point_str = line_split[:response_col] + line_split[(response_col+1):]
                point_float = [float(component) for component in point_str]
                all_points.append(point_float)
            
    return all_points

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

def exp_mu(input, mu, theta=1):
    """
    Exponential function on ||input - mu||^2,
    mu and x are lists with same length
    """
    assert(len(input) == len(mu))
    square_mu_input = [(xi - mu_i)**2 for xi, mu_i in zip(input, mu)]
    norm2 = sum(square_mu_input)

    exp_mu = np.exp(-0.5 * norm2 / theta**2)

    return exp_mu

def rkhs_norm(lst_coeff, lst_mu, theta=1):
    """
    compute the RKHS norm  
    """
    cross_part = 0
    for ind_a, coeff_a in enumerate(lst_coeff):
        for ind_b, coeff_b in enumerate(lst_coeff):
            cross_part += coeff_a * coeff_b * exp_mu(lst_mu[ind_a], lst_mu[ind_b], theta=theta)
    
    norm = np.sqrt(cross_part)

    return norm

def diff_mu1_mu2(mu1, mu2, theta=1):
    "The RKHS norm of exp_mu(,mu1) - exp_mu(,mu2)"
    norm_rkhs = np.sqrt(2 - 2 * exp_mu(mu1, mu2, theta))

    return norm_rkhs

def read_points_from_file(file_path):  
    points = []  
    
    with open(file_path, 'r') as file:  
        # 跳过文件头  
        next(file)  
        
        for line in file:  
            # 分割每一行  
            values = line.split()  
            # 将响应值（函数值）和点的坐标存储  
            response_value = float(values[0])  
            coordinates = [float(value) for value in values[1:]]  
            points.append((response_value, coordinates))  
    
    return points  

def find_max_points(file_path, r):  
    points = read_points_from_file(file_path)  
    
    # 将 (函数值, 点坐标) 分离，方便处理  
    values, coordinates = zip(*points)  
    coordinates = np.array(coordinates)  
    
    found_points = []  
    
    # 找到最大值点  
    max_value_index = np.argmax(values)  
    max_point = coordinates[max_value_index]  
    found_points.append(max_point)  
    
    while True:  
        max_value = float('-inf')  
        new_point = None  
        
        for i, point in enumerate(coordinates):  
            
            # 检查点是否在已找到点的半径范围外  
            is_outside = True  
            for found_point in found_points:  
                if np.linalg.norm(point - found_point) <= r:  
                    is_outside = False  
                    break  
            
            if is_outside:  
                value = values[i]  
                if value > max_value:  
                    max_value = value  
                    new_point = point  
        
        if new_point is None:  
            # 没有找到新点，结束循环  
            break   
        
        found_points.append(new_point)  
    
    return found_points  

def get_params_target(file_readme):
    "get target parameters from readme file"
    with open(file_readme, 'r', encoding="utf-8") as f_r:
        for line in f_r:
            if ("Similarity" not in line) and len(line) > 2:
                line = line.strip()
                line_split_unnormalized = line.split('#')
                line_split_tmp = [ele.strip('[') for ele in line_split_unnormalized]
                line_split = [ele.strip(']') for ele in line_split_tmp]

                alpha1 = line_split[1]
                alpha2 = line_split[2]
                alpha3 = line_split[3]
                
                beta1_tmp = line_split[4]
                beta1 = [float(ele.strip()) for ele in beta1_tmp.split(',')]

                beta2_tmp = line_split[5]
                beta2 = [float(ele.strip()) for ele in beta2_tmp.split(',')]

                beta3_tmp = line_split[6]
                beta3 = [float(ele.strip()) for ele in beta3_tmp.split(',')]

    paramters_target = {
        "alpha1": float(alpha1),
        "beta1":  [beta1], 
        "alpha2": float(alpha2),
        "beta2":  [beta2],
        "alpha3": float(alpha3),
        "beta3":  [beta3]
        }
    
    return paramters_target


# if __name__ == "__main__":
#     l_bounds = [1, 3]
#     u_bounds = [5, 9]
#     print(check_inBounds([-3, 5], l_bounds, u_bounds))

#     file_dir = "./data/2D_Triple2Triple_sample_backup/2D_Triple2Triple_sample_bad_prior_scaleTheta/20"
#     file_name = "simTriple2Triple2D_points_task0_sample.tsv"
#     draw_2d_lhd(file_sampling=file_dir+'/'+file_name)

#     # Maxmin LHD illustration    
#     xlimits = np.array([[1.0, 4.0], [2.0, 3.0]])
#     sampling = LHS(xlimits=xlimits, criterion='maximin')

#     num = 10
#     x = sampling(num)

#     print(x.shape)

#     plt.plot(x[:, 0], x[:, 1], "o")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.show()

