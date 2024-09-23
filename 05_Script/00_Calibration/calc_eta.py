# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 23:41:41 2024

@author: hirar
"""

base_path = "../../04_Data/00_Calibration/swing/"
file_names = ['LOG00014.TXT', 'LOG00015.TXT', 'LOG00016.TXT']

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import decoder
from scipy import interpolate

A_INTER = -0.0803317
B_INTER = 120.248

A_OUTER = -0.0835924
B_OUTER = 203.251


def get_inter_angle(val):
    return A_INTER*val + B_INTER

def get_outer_angle(val):
    return A_OUTER*val + B_OUTER

class lowpass:
    def __init__(self, k, x0):
        self.k = k
        self.zx = x0
    
    def update(self, x):
        temp_x = self.k * x + (1-self.k) * self.zx
        self.zx = temp_x
        return temp_x

def damp_func(t, X0, xi, a, omega, t0, s0):
    return X0*(1-a*(t-t0)) * np.exp(-xi* omega *(t-t0)) * np.cos(omega * (t-t0) + s0)


def fit_func(t, xi, a):
    return (1-a*t) * np.exp(-xi *t)

class DampData:
    def __init__(self, time_array, data_array, t0, t1):
        self.i_range = np.where((time_array>t0) & (time_array<t1))
        self.t_raw = time_array[self.i_range] - t0
        self.data_raw = data_array[self.i_range]
        self.data_lowpass = data_array[self.i_range]
        self.data_func = interpolate.interp1d(self.t_raw, self.data_raw)
        self.data_len = len(self.t_raw)    

        self.diff = np.array([1]*self.data_len)
        self.upper_peak = []
        self.lower_peak = []        
        
    def plot(self):
        if self.is_upper_start == True:
            #plt.plot(self.t_raw, self.data_raw)
            plt.plot(self.t_raw, self.data_lowpass)
            plt.plot(self.t_upper_peak, self.upper_peak, 'ro')
            plt.plot(self.t_lower_peak, self.lower_peak, 'go')
            plt.plot(self.t_raw, damp_func(self.t_raw, self.X0, self.xi_fit, self.a, self.omega, self.t0, 0), 'k--')
            plt.plot(self.t_raw, self.X0* fit_func(self.t_raw, self.xi_fit*self.omega , self.a), 'k--')
            
        else:
            #plt.plot(self.t_raw, -self.data_raw)
            plt.plot(self.t_raw, -self.data_lowpass)
            plt.plot(self.t_upper_peak, -self.upper_peak, 'ro')
            plt.plot(self.t_lower_peak, -self.lower_peak, 'go')
            plt.plot(self.t_raw, -damp_func(self.t_raw, self.X0, self.xi_fit, self.a, self.omega, self.t0, np.pi), 'k--')
            plt.plot(self.t_raw, self.X0* fit_func(self.t_raw, self.xi_fit*self.omega , self.a), 'k--')
        
        plt.ylim([-40, 40])
        
    
    def set_lopass(self, k):
        temp_list = []
        temp_lowpass = lowpass(k, self.data_raw[0])
        for data in self.data_raw:
            temp_list.append(temp_lowpass.update(data))
        self.data_lowpass = np.array(temp_list)

    def check_peak(self, spt_num):          
        diff_list = [1,1,1]
        i_upper_peak = []
        i_lower_peak = []
        is_lower_peak = False
        spt_sw = False
        spt_cnt = 0
        i_peak = 0
        i = 3
        while i < self.data_len-3:
            f0 = self.data_lowpass[i-3]
            f1 = self.data_lowpass[i-2]
            f2 = self.data_lowpass[i-1]
            f4 = self.data_lowpass[i+1]
            f5 = self.data_lowpass[i+2]
            f6 = self.data_lowpass[i+3]
            f_dash = (-f0 +9*f1 -45*f2 +45*f4 -9*f5 +f6)
            
            if f_dash > 0:
                diff = 1
            else:
                diff = -1
            diff_list.append(diff)
            
            if diff*diff_list[i-1] < 0:
                if diff > 0:
                    is_lower_peak = True # 負方向から正方向なので下側
                else:
                    is_lower_peak = False # 正方向から負方向なので上側
                spt_sw = True
                i_peak = i
                spt_cnt = 0
            else:
                spt_cnt += 1
            
            if spt_sw == True:
                if spt_cnt > spt_num:
                    if is_lower_peak == True:
                        i_lower_peak.append(i_peak)
                    else:
                        i_upper_peak.append(i_peak)
                    spt_cnt = 0
                    spt_sw = False
                    
                
            i += 1
        self.diff = diff
        self.i_upper_peak = i_upper_peak
        self.i_lower_peak = i_lower_peak
        self.upper_peak = self.data_lowpass[i_upper_peak]
        self.lower_peak = self.data_lowpass[i_lower_peak]
        self.t_upper_peak = self.t_raw[i_upper_peak]
        self.t_lower_peak = self.t_raw[i_lower_peak]
        self.X0 = max(np.abs(max(self.upper_peak)), np.abs(min(self.lower_peak)))
        self.t0 = min(min(self.t_upper_peak), min(self.t_lower_peak))
        if self.data_func(self.t0) > 0:
            self.is_upper_start = True
        else:
            self.is_upper_start = False
        
    def get_eta(self, num_T):
        if len(self.upper_peak)>num_T and len(self.lower_peak) > num_T:
            upper_delta = np.log(self.upper_peak[0]/self.upper_peak[num_T]) /num_T
            lower_delta = np.log(self.lower_peak[0]/self.lower_peak[num_T]) /num_T
            upper_T = (self.t_upper_peak[num_T] - self.t_upper_peak[0])/num_T
            lower_T = (self.t_lower_peak[num_T] - self.t_lower_peak[0])/num_T
            
            self.eta = ((upper_delta + lower_delta)/2) / np.pi
            self.xi = self.eta/2
            self.T = (upper_T + lower_T)/2
            self.omega = 2*np.pi / self.T

        else:
            self.eta = 0
            self.T = 1    
        
        if self.is_upper_start == True:
            opt, cov = curve_fit(fit_func, self.t_upper_peak, self.upper_peak/self.X0, p0 = [0.01*self.omega, 0.07], bounds=([0,0], [1, 1]))
        
        else:
            opt, cov = curve_fit(fit_func, self.t_lower_peak, -self.lower_peak/self.X0, p0 = [0.01*self.omega, 0.07], bounds=([0,0], [1, 1]))
        
        self.xi_fit = opt[0]/self.omega
        self.eta_fit = self.xi_fit*2
        self.a = opt[1]
        
        
        
        





"""
# Inter
file_path = base_path + file_names[0]
time, inter_val, outer_val, gps = decoder.check_file(file_path)
st_list = [2.04, 4.705, 7.02, 9.325]
ed_list = [3.2, 5.905, 8.2, 10.525]
angle_data = get_inter_angle(inter_val)
#plt.plot(time, get_inter_angle(inter_val))
offset = 0.5
"""

"""
# Outer
file_path = base_path + file_names[1]
time, inter_val, outer_val, gps = decoder.check_file(file_path)
st_list = [1.105, 6.53, 15.46]
ed_list = [2.72, 8.12, 17.18]
#st_list = [1.105, 6.53]
#ed_list = [2.72, 8.12]
angle_data = get_outer_angle(outer_val)
offset = 3.5
#plt.plot(time, get_outer_angle(outer_val))



i = 0
Data = []

while i < len(st_list):
    temp_data = DampData(time, angle_data-offset, st_list[i], ed_list[i])
    temp_data.set_lopass(0.02)
    temp_data.check_peak(50)
    temp_data.get_eta(1)
    temp_data.plot()
    #plt.plot(temp_data.t_upper_peak - temp_data.t0, temp_data.upper_peak, 'o')
    #plt.plot(temp_data.t_lower_peak - temp_data.t0 + temp_data.T/4, temp_data.lower_peak, 'o')
    Data.append(temp_data)
    i += 1


xi_list = []
a_list = []
omega_list = []

for data in Data:
    xi_list.append(data.xi_fit)
    a_list.append(data.a)
    omega_list.append(data.omega)

xi_list = np.array(xi_list)
a_list = np.array(a_list)
omega_list = np.array(omega_list)

xi_ave = np.average(xi_list)
a_ave = np.average(a_list)
omega_ave = np.average(omega_list)
    
print(xi_ave, a_ave, omega_ave)

#data = Data[2]
#data.plot()
#plt.plot(data.t_raw, data.X0* fit_func(data.t_raw, opt[0], opt[1]), 'k--')

"""


#closs validation
file_path = base_path + file_names[2]
time, inter_val, outer_val, gps = decoder.check_file(file_path)

xi_inter = 0.01402669
xi_outer = 0.01048767
a_inter = 0.69339607
a_outer = 0.50631147
omega_inter = 23.1889512
omega_outer = 23.4509134

st_list = [1.2, 3.27,  12.39]
ed_list = [2.55, 4.72, 13.5]
angle_inter = get_inter_angle(inter_val)
angle_outer = get_outer_angle(outer_val)
offset_inter = 0.5
offset_outer = 3.5
#plt.plot(time, get_outer_angle(outer_val))
cross_i_list = []
cross_o_list = []
i = 0
while i < len(st_list):
    temp_cross_i = DampData(time, angle_inter - offset_inter, st_list[i], ed_list[i])
    temp_cross_o = DampData(time, angle_outer - offset_outer, st_list[i], ed_list[i])
    temp_cross_i.set_lopass(0.04)
    temp_cross_i.check_peak(50)
    temp_cross_i.get_eta(2)
    temp_cross_i.xi_fit = xi_inter
    temp_cross_i.a = a_inter
    temp_cross_i.omega = omega_inter
    
    temp_cross_o.set_lopass(0.04)
    temp_cross_o.check_peak(50)
    temp_cross_o.get_eta(2)
    temp_cross_o.xi_fit = xi_outer
    temp_cross_o.a = a_outer
    temp_cross_o.omega = omega_outer
    cross_i_list.append(temp_cross_i)
    cross_o_list.append(temp_cross_o)
    
    i += 1

num = 2
cross_o_list[num].plot()
cross_i_list[num].plot()

