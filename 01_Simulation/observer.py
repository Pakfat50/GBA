# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:28:03 2024

@author: hirar
"""

import numpy as np
from matplotlib import pyplot as plt

Easy = False

if Easy == True:

    m = 0.5
    l = 0.1
    g = 9.81
    I = m*l**2
    eta = 0.01
    dt = 0.0001
    a_21 = -m*g*l/I
    a_22 = -eta/I
    
    x = np.array([np.radians(5),0])
    x_array = [x]
    t_array = np.arange(0,10,dt)
    
    i = 1
    
    while i < len(t_array):
        z_x = x_array[i-1]
        z_theta = z_x[0]
        z_theta_dot = z_x[1]
        
        #シミュレーター更新
        theta =  z_theta_dot* dt + z_theta
        theta_dot = (a_21*np.sin(z_theta) + a_22*z_theta_dot)*dt + z_theta_dot
        temp_x = np.array([theta, theta_dot])
        x_array.append(temp_x)
        i += 1    
    
    x_array = np.array(x_array)
    plt.plot(t_array, x_array[:,0], 'r')
    plt.plot(t_array, x_array[:,1], 'b')


else:
    m_lod = 0 # [kg] #ロッドの重量
    l1 = 0 # [m] 支点から球までの長さ
    l2 = 0.1 # [m] 支点から錘までの長さ
    m_bal = 0 # [kg] 球の質量
    m_w = 0.5 # [kg] #錘の質量
    eta = 0.01 # 減衰率
    dt = 0.001 # [sec] タイムステップ
    g = 9.81 # [m/sec^2] 重力加速度
    
    L_lod = l1 + l2
    r_lod = abs(abs(l1-l2)-L_lod/2)
    I_lod = (m_lod*L_lod**2)/12 + m_lod*r_lod**2
    I_tot = I_lod + m_bal*l1**2 + m_w*l2**2
    
    a_21 = -((m_w*l2) - (m_bal*l1))*g / I_tot
    a_22 = - eta/ I_tot
    a_23 = - l2/ I_tot
    
    
    x = np.array([np.radians(5),0,0])   # Θ[deg], Θ_dot[deg/sec], f[N]
    x_hat = np.array([0,0,0]) #推定値 Θ[deg], Θ_dot[deg/sec], f[N]
    L = np.array([-0.01, 0, -9]) #オブザーバーゲイン
    
    def f_func(t):
        return 0.5 + np.sin(t) + 0.5*np.sin(t/3) + 0.2*np.sin(t/2-0.4)

    
    x_array = [x]
    x_hat_array = [x_hat]
    
    t_array = np.arange(0,100, dt)
    
    i = 1
    
    while i < len(t_array):
        z_x_hat = x_hat_array[i-1]
        z_theta_hat = z_x_hat[0]
        z_theta_dot_hat = z_x_hat[1]
        z_f_hat = z_x_hat[2]
        
        z_x = x_array[i-1]
        z_theta = z_x[0]
        z_theta_dot = z_x[1]
        z_f = z_x[2]
        
        #オブザーバー更新
        theta_hat = (L[0] * (z_theta - z_theta_hat) + z_theta_dot_hat) * dt + z_theta_hat
        theta_dot_hat = (a_21 * np.sin(z_theta_hat) + L[1] * (z_theta - z_theta_hat) + a_22*z_theta_dot_hat + a_23*z_f_hat*np.cos(z_theta_hat))*dt + z_theta_dot_hat
        f_hat = (L[2] * (z_theta - z_theta_hat)) * dt + z_f_hat
        temp_x_hat = np.array([theta_hat, theta_dot_hat, f_hat])
        x_hat_array.append(temp_x_hat)
        
        #シミュレーター更新
        theta =  z_theta_dot* dt + z_theta
        theta_dot = (a_21*np.sin(z_theta) + a_22*z_theta_dot + a_23*z_f*np.cos(z_theta))*dt + z_theta_dot
        f = f_func(t_array[i])
        temp_x = np.array([theta, theta_dot, f])
        x_array.append(temp_x)
        i += 1
        
    
    x_array = np.array(x_array)
    x_hat_array = np.array(x_hat_array)
    
    plt.plot(t_array, x_hat_array[:,2], 'b')
    plt.plot(t_array, x_array[:,2], 'r--')
