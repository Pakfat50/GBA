# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:28:03 2024

@author: hirar
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate as intp

Easy = False
Easy_mdl = False
ArrayMod = True
NonLinear = True
AddNoise = True
UseSensor = True
AngularRateSens = True
LowpassEna = True
WindMode = 1 #1：波の重ね合わせ、2：t=5からのステップ波、3：0.01sec、20m/sの疑似インパルス波


WindGain = 5
WindOffset = 1
LowpassGain = 0.2
LowpassAngRateGain = 1
LowpssFGain = 1

NoiseThetaGain = 0.5 #[deg]
NoiseThetaDotGain = 1 #[deg/sec]
S_ball = 0.15**2/4 * np.pi
cd = 0.3
rho = 1.225
TMIN = 0
TMAX = 30
dt_sim = 0.001 # [sec] タイムステップ
dt_obs = 0.02 # [sec] タイムステップ
g = 9.81 # [m/sec^2] 重力加速度
    

class lowpass:
    def __init__(self, k, x0):
        self.k = k
        self.zx = x0
    def update(self, x):
        temp_x = self.k * x + (1-self.k) * self.zx
        self.zx = temp_x
        return temp_x
    

if Easy == True:

    m = 0.5
    l = 0.1
    I = m*l**2
    eta = 0.01
    a_21 = -m*g*l/I
    a_22 = -eta/I
    
    x = np.array([np.radians(5),0])
    x_array = [x]
    t_array = np.arange(0,10,dt_sim)
    
    i = 1
    
    while i < len(t_array):
        z_x = x_array[i-1]
        z_theta = z_x[0]
        z_theta_dot = z_x[1]
        
        #シミュレーター更新
        theta =  z_theta_dot* dt_sim + z_theta
        theta_dot = (a_21*np.sin(z_theta) + a_22*z_theta_dot)*dt_sim + z_theta_dot
        temp_x = np.array([theta, theta_dot])
        x_array.append(temp_x)
        i += 1    
    
    x_array = np.array(x_array)
    plt.plot(t_array, x_array[:,0], 'b')
    
else:
    if Easy_mdl == True:
        m_lod = 0 # [kg] #ロッドの重量
        l1 = 0.1 # [m] 支点から球までの長さ
        l2 = 0.1 # [m] 支点から錘までの長さ
        m_bal = 0 # [kg] 球の質量
        m_w = 0.5 # [kg] #錘の質量
        eta = 0.01 # 減衰率
   
    else: 
        m_lod = 0.015 # [kg] #ロッドの重量
        l1 = 0.1 # [m] 支点から球までの長さ
        l2 = 0.3 # [m] 支点から錘までの長さ
        m_bal = 0.015 # [kg] 球の質量
        m_w = 0.03 # [kg] #錘の質量
        eta = 0.01 # 減衰率

    L_lod = l1 + l2
    r_lod = abs(abs(l1-l2)-L_lod/2)
    I_lod = (m_lod*L_lod**2)/12 + m_lod*r_lod**2
    I_tot = I_lod + m_bal*l1**2 + m_w*l2**2
    
    a_21 = -((m_w*l2) - (m_bal*l1))*g / I_tot
    a_22 = - eta/ I_tot
    a_23 = - l2/ I_tot
    
    
    # オブザーバーゲイン設計
    # https://qiita.com/trami/items/02f24e8eb68c05da4b1b

    # オブザーバの極
    # ---------------------------------------------------------
    #new_pole = [-5, -20, -12]
    new_pole = [-10, -30, -12]
    # ---------------------------------------------------------
    
    # 状態空間表現
    # ---------------------------------------------------------
    A = np.array([[0,    1,    0   ],\
                 [a_21, a_22, a_23],\
                 [0,    0,    0   ]])
    c = np.array([[1, 0, 0]])
    
    dim = np.shape(A)[0]
    print("A = \n"+ str(A) + "\n")
    print("c = \n"+ str(c) + "\n")
    # ---------------------------------------------------------
    
    # 固有多項式の係数を求める
    # ---------------------------------------------------------
    eig = np.linalg.eigvals(A)
    sI_A = np.poly1d(eig, True, variable='s')
    # print('sI_A = \n' + str(sI_A) + "\n")
    
    print(np.array(sI_A))
    alpha = np.round(sI_A)
    alpha = alpha[::-1]
    print('alpha = \n' + str(alpha) + "\n")
    # ---------------------------------------------------------
    
    # 可観測行列
    # ---------------------------------------------------------
    Uo = np.array(c)
    for i in range(dim - 1):
        if i == 0:
            temp =c
        else:
            temp = Uo[i,:]
        # print('temp =' +str(temp) + "\n")
        # print('Uc =' +str(Uo) + "\n")
        Uo = np.vstack((Uo, np.dot(temp, A)))
    
    print("Uo = \n"+ str(Uo) + "\n")
    
    rank_Uo = np.linalg.matrix_rank(Uo)
    print("rank_U0 = " + str(rank_Uo) + "\n")
    # ---------------------------------------------------------
    
    # 行列Vを求める
    # ---------------------------------------------------------
    V = np.zeros((dim, dim))
    for y in range(dim):
        for x in range(dim):
            if x + y < dim - 1:
                V[y,x] = alpha[x+y+1]
            elif x + y == dim - 1:
                V[y,x] = 1
    
    print("V = \n"+ str(V) + "\n")
    # ---------------------------------------------------------
    
    
    # 変換行列Tを求める
    # ---------------------------------------------------------
    To = np.dot(V, Uo)
    print("To = VUo = \n"+ str(To) + "\n")
    To_inv = np.linalg.inv(To)
    print("To^-1 = (VUo) = \n"+ str(To_inv) + "\n")
    # ---------------------------------------------------------
    
    # オブザーバゲインを求める
    # ---------------------------------------------------------
    p_s = np.poly1d(new_pole, True, variable='s')
    # print('p_s = \n' + str(p_s) + "\n")
    
    beta = np.round(np.array(p_s))
    beta = beta[::-1]
    # print('beta = \n' + str(beta) + "\n")
    
    a_b = alpha-beta
    a_b = np.delete(a_b, dim, 0)
    a_b = np.reshape(a_b, (dim, 1))
    # print('a_b = \n' + str(a_b) + "\n")
    
    L = np.dot(To_inv, a_b)
    print("L = \n"+ str(L) + "\n")
    # ---------------------------------------------------------


    u_array = [0]
    u_est_array = [0]
    x = np.array([np.radians(0),0,0])   # Θ[deg], Θ_dot[deg/sec], f[N]
    x_hat = np.array([0,0,0]) #推定値 Θ[deg], Θ_dot[deg/sec], f[N]
    #L = np.array([[-34], [-274], [13.01883333]]) #オブザーバーゲイン

    #風のモデル化
    #波の作成 https://qiita.com/mumei_456/items/edf32d67d83f423df87d

    num_waves = 11 #波の数
    frequency = [0.005, 0.01, 0.1, 0.5, 1,    2,    3,    4,    5,     10,    20]
    amplitude = [0,1,   0.5,  0.2, 0.2, 0.01, 0.01, 0.01, 0.01, 0.025, 0.025, 0.001]
    amplitude/= np.sum(amplitude)
    phase = []
    #ランダムな位相の波の生成
    for i in range(num_waves):
        phase.append(np.random.uniform(-np.pi , np.pi)) #位相
    
    def u_func(t):
        if WindMode == 1:
            makewave = 0
            for i in range(num_waves):
                temp_phase = np.random.uniform(-np.pi/20 , np.pi/20) + phase[i]
                wave = amplitude[i] * np.sin(2 * np.pi * frequency[i] * t + temp_phase) * WindGain
                makewave += wave #合成波の生成
                U = makewave + WindOffset
        elif WindMode == 2:
            if t < 5:
                U = 0
            elif (t >= 5) and (t < 20):
                U = WindGain
            else:
                U = 0
        else:
            if t < 5:
                U = 0
            elif (t >= 5) and (t < 5.01):
                U = 20
            else:
                U = 0
        return U
        
    
    def f_func(u, theta, theta_dot):
        u_rot = l1*theta_dot*np.cos(theta)
        u_rel = u-u_rot
        if u_rel > 0:
            direction = 1
        else:
            direction = -1
        return 0.5 * rho * u_rel**2 * cd * S_ball * direction
    
    def getWindSpeed(theta, theta_dot, f):
        if f > 0:
            direction = 1
        else:
            direction = -1
        u_uni = np.sqrt(2*np.abs(f)/(S_ball*cd*rho))*direction
        u_rot = l1 * theta_dot*np.cos(theta)
        u = u_uni + u_rot
        return u


    #simulation
    x_array = [x]
    t_sim_array = np.arange(TMIN,TMAX, dt_sim)
    
    i = 1
    
    while i < len(t_sim_array):
        temp_u = u_func(t_sim_array[i])
        u_array.append(temp_u)
        temp_f = f_func(temp_u, x_array[i-1][0], x_array[i-1][1])
        
        if ArrayMod == True:
            zx = x_array[i-1]
            zy = np.dot(c, zx)
            
            x_dot = np.dot(A, zx)
            x = x_dot*dt_sim + zx
            x[2] = temp_f
            x_array.append(x)
            i += 1
        else:
            z_x = x_array[i-1]
            z_theta = z_x[0]
            z_theta_dot = z_x[1]
            z_f = z_x[2]
            
            theta =  z_theta_dot* dt_sim + z_theta
            f = temp_f
            
            if NonLinear == True:
                theta_dot = (a_21*np.sin(z_theta) + a_22*z_theta_dot + a_23*z_f*np.cos(z_theta))*dt_sim + z_theta_dot    
            else:
                theta_dot = (a_21*z_theta + a_22*z_theta_dot + a_23*z_f)*dt_sim + z_theta_dot


            temp_x = np.array([theta, theta_dot, f])            
            x_array.append(temp_x)   
            i += 1
    x_array = np.array(x_array)


    #observer
    theta_func = intp.interp1d(t_sim_array, x_array[:,0])
    theta_dot_func = intp.interp1d(t_sim_array, x_array[:,1])
    f_sim_func = intp.interp1d(t_sim_array, x_array[:,2])
    theta_sense_array = [0]
    theta_dot_sence_array = [0]
    
    theta_lowp = lowpass(LowpassGain, 0)
    theta_dot_lowp = lowpass(LowpassAngRateGain, 0)
    f_lowp = lowpass(LowpssFGain, 0)
    
    x_hat_array = [x_hat]
    t_obs_array = np.arange(TMIN,TMAX, dt_obs)

    i = 1
    
    while i < len(t_obs_array):
        t_obs = t_obs_array[i]
        temp_theta = theta_func(t_obs)
        temp_theta_dot = theta_dot_func(t_obs)
        temp_f = f_sim_func(t_obs)
        zx = np.array([temp_theta, temp_theta_dot, temp_f])
        sensingNoise = np.array([np.radians(np.random.uniform(-NoiseThetaGain, NoiseThetaGain)), \
                                 np.radians(np.random.uniform(-NoiseThetaDotGain, NoiseThetaDotGain)), \
                                 0])
        
        if i > 5:
            x4 = theta_dot_sence_array[i-4]
            x3 = theta_dot_sence_array[i-3]
            x2 = theta_dot_sence_array[i-2]
            x1 = theta_dot_sence_array[i-1]
            temp_theta_dot_sence = (x4 -8*x3 + 8*x2 -x1)/(12*t_obs)
            
            if LowpassEna == True:
                temp_theta_dot_sence = theta_dot_lowp.update(temp_theta_dot_sence)
                
            theta_dot_sence_array.append(temp_theta_dot_sence)
        
        
        else:
            temp_theta_dot_sence = 0
            theta_dot_sence_array.append(temp_theta_dot_sence)
            
        
        if AddNoise == True:
            zx = zx + sensingNoise
            if LowpassEna == True:
                zx[0] = theta_lowp.update(zx[0])
            
        if ArrayMod == True:
            zx_hat = x_hat_array[i-1]
            zy = np.dot(c, zx)
            x_hat_dot = np.dot((A + np.dot(L, c)), zx_hat) - np.dot(L, zy)
            x_hat = x_hat_dot*dt_obs + zx_hat
            x_hat_array.append(x_hat)
            
        else:
            z_theta = zx[0]
            z_x_hat = x_hat_array[i-1]
            z_theta_hat = z_x_hat[0]
            z_theta_dot_hat = z_x_hat[1]
            z_f_hat = z_x_hat[2]

            #オブザーバー更新
            theta_hat = (L[0][0]*(z_theta_hat - z_theta) + z_theta_dot_hat)*dt_obs + z_theta_hat
            f_hat = (L[2][0]*(z_theta_hat - z_theta))*dt_obs + z_f_hat

            
            if NonLinear == True:
                #オブザーバー更新
                theta_dot_hat = (a_21*np.sin(z_theta_hat) + L[1][0]*(z_theta_hat - z_theta) + a_22*z_theta_dot_hat + a_23*z_f_hat*np.cos(z_theta_hat))*dt_obs + z_theta_dot_hat
            else:
                #オブザーバー更新
                theta_dot_hat = (a_21*z_theta_hat + L[1][0]*(z_theta_hat - z_theta) + a_22*z_theta_dot_hat + a_23*z_f_hat)*dt_obs + z_theta_dot_hat


            x_hat = np.array([theta_hat, theta_dot_hat, f_hat])         
            x_hat_array.append(x_hat)
                    
        if LowpassEna == True:
            temp_f_lowp = f_lowp.update(x_hat[2])
        theta_sense_array.append(zx[0])
        u_est_array.append(getWindSpeed(x_hat[0], temp_theta_dot_sence, temp_f_lowp))
        i += 1    
            
    x_hat_array = np.array(x_hat_array)
    
    
    fig = plt.figure(figsize=(9,12))
    fig.suptitle('Noise Gain:%s[deg], dt_sim:%s[sec], dt_obs:%s[sec]'%(NoiseThetaGain, dt_sim, dt_obs))
    
    ax1 = fig.add_subplot(4,1,1)
    ax1.plot(t_sim_array, u_array, 'b')
    ax1.plot(t_obs_array, u_est_array, 'r--')
    ax1.set_xlabel('Time[sec]')
    ax1.set_ylabel('Input Wind Speed[m/s]')
    ax1.legend(['Simulator', 'Estimate'])
    
    ax2 = fig.add_subplot(4,1,2)
    ax2.plot(t_sim_array, np.degrees(x_array[:,0]), 'b')
    ax2.plot(t_obs_array, np.degrees(x_hat_array[:,0]), 'r--')
    ax2.plot(t_obs_array, np.degrees(theta_sense_array), 'g--')
    ax2.set_xlabel('Time[sec]')
    ax2.set_ylabel('Angle[deg]')
    ax2.legend(['Simulator', 'Observer', 'SensorModel'])
    
    ax3 = fig.add_subplot(4,1,3)
    ax3.plot(t_sim_array, np.degrees(x_array[:,1]), 'b')
    ax3.plot(t_obs_array, np.degrees(x_hat_array[:,1]), 'r--')
    ax3.set_xlabel('Time[sec]')
    ax3.set_ylabel('Angular Rate[deg/sec]')
    ax3.legend(['Simulator', 'Observer'])
    
    ax4 = fig.add_subplot(4,1,4)
    ax4.plot(t_sim_array, x_array[:,2], 'b')    
    ax4.plot(t_obs_array, x_hat_array[:,2], 'r--')
    ax4.set_xlabel('Time[sec]')
    ax4.set_ylabel('Disturbance Force[N]')
    ax4.legend(['Simulator', 'Observer'])
    
    fig.tight_layout()
    plt.show()