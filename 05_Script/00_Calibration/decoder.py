# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:51:46 2024

@author: hirar
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit  


def linear_func(x, a, b):
    return a*x + b


def get_ave_val(base_path, file_list):
    ave_list = []
    i = 0
    while i < len(file_list):
        filepath = base_path + file_list[i]
        
        time, inter_val, outer_val, gps = check_file(filepath)
        
        inter_axis_val = np.average(inter_val.astype(float))
        outer_axis_val = np.average(outer_val.astype(float))
        
        ave_list.append([inter_axis_val, outer_axis_val])
        i += 1
    
    return np.array(ave_list)


def check_file(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    
    time_list = []
    inter_val_list = []
    outer_val_list = []
    gps_list = []
    
    for line in lines:
        datas = line.split('\t')
        if len(datas) == 4:
            try:
                time_list.append(float(datas[0])/1000000)
                inter_val_list.append(float(datas[2]))
                outer_val_list.append(float(datas[1]))
                gps_list.append(str(datas[3]))
            except:
                print(line)
                
        else:
            print(line)
    return np.array(time_list), np.array(inter_val_list), np.array(outer_val_list), np.array(gps_list)


if __name__ == '__main__':     
    inter_base_path = "../../04_Data/00_Calibration/InterAxis/"
    outer_base_path = "../../04_Data/00_Calibration/OuterAxis/"
    
    inter_angles = [-30, -20, -10, 10, 20, 30]
    outer_angles = [-30, -20, -10, 10, 20, 30]
    
    inter_files = ["LOG00009.TXT", "LOG00008.TXT", "LOG00007.TXT", "LOG00006.TXT", "LOG00005.TXT", "LOG00004.TXT"]
    outer_files = ["LOG00014.TXT", "LOG00013.TXT", "LOG00012.TXT", "LOG00011.TXT", "LOG00010.TXT", "LOG00009.TXT"]

    
    inter_ave_array = get_ave_val(inter_base_path, inter_files)
    outer_ave_array = get_ave_val(outer_base_path, outer_files)
    
    
    inter_opt, inter_cov = curve_fit(linear_func, inter_ave_array[:,0], inter_angles)
    outer_opt, outer_cov = curve_fit(linear_func, outer_ave_array[:,1], outer_angles)
    
    
    plt.plot(inter_ave_array[:,0], inter_angles, 'bo-')
    plt.plot(inter_ave_array[:,0], linear_func(inter_ave_array[:,0], inter_opt[0], inter_opt[1]), 'b--')
    
    plt.plot(outer_ave_array[:,1], outer_angles, 'ro-')
    plt.plot(outer_ave_array[:,1], linear_func(outer_ave_array[:,1], outer_opt[0], outer_opt[1]), 'r--')
    
    print(inter_opt)
    print(outer_opt)

