# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 22:45:07 2024

@author: hirar
"""

import numpy as np

g = 9.81

class simulator:
    def __init__(self, I, eta, M, lg, lf, T_loss):
        self.I = I
        self.eta = eta
        self.M = M
        self.lg = lg
        self.lf = lf
        self.T_loss = T_loss
        self.theta = 0
        self.theta_dot = 0
        self.f = 0
    
    def set_init_val(self, theta0, theta_dot0, f0):
        self.theta = theta0
        self.theta_dot = theta_dot0
        self.f = f0
        
    def update(self, f, dt):
        if self.z_theta_dot > 0:
            sgn = 1
        else:
            sgn = -1
        theta_dot_dot = (self.M*self.lg*g*self.theta)/self.I - self.eta/self.I - (self.lb * f)/self.I - (sgn*self.T_loss)/self.I
        theta_dot = theta_dot_dot * dt + self.theta_dot
        theta =  theta_dot* dt + self.theta   
        
        self.theta = theta
        self.theta_dot = theta_dot
        self.f = f
        
        
class observer:
    def __init__(self, I, eta, M, lg, lf, T_loss):
        self.I = I
        self.eta = eta
        self.M = M
        self.lg = lg
        self.lf = lf
        self.T_loss = T_loss
        self.theta_hat = 0
        self.theta_dot_hat = 0
        self.f_hat = 0
        self.k1 = 1
        self.k2 = 1
        self.k3 = 1
    
    def set_init_val(self, theta0, theta_dot0, f0):
        self.theta_hat = theta0
        self.theta_dot_hat = theta_dot0
        self.f_hat = f0
        
    def set_gain(self, k1, k2, k3):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3        
        
    def update(self, theta, dt):
        if self.z_theta_dot > 0:
            sgn = 1
        else:
            sgn = -1
        theta_dot_dot_hat = (self.M*self.lg*g*self.theta_hat)/self.I - self.eta/self.I - (self.lb * self.f_hat)/self.I - (sgn*self.T_loss)/self.I
        theta_dot_hat = (theta_dot_dot_hat + self.k2 * (theta - self.theta_hat))* dt + self.theta_dot_hat 
        theta_hat = (theta_dot_hat + self.k1*(theta - self.theta_hat))* dt + self.theta_hat    
        f_hat = (self.k3*(theta- self.theta_hat))*dt + self.f_hat
        
        self.theta_hat = theta_hat
        self.theta_dot_hat = theta_dot_hat
        self.f_hat = f_hat