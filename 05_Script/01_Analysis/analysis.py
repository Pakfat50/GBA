# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:56:47 2025

@author: hirar
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate as intp
import datetime


S_ball = 0.15**2/4 * np.pi
cd = 0.55
rho = 1.225
g = 9.81 # [m/sec^2] 重力加速度
l1 = 0.245 # [m] 支点から球までの長さ
l2 = 0.18 # [m] 支点から錘までの長さ
m = 5.872 * 10**-3 # [kg] 錘の等価質量
k = l2/l1 * m * g # [N] 錘の角度から空気力を算出する係数。モーメントのつり合いから

rot_Offset = 165


def lowpass(data, k):
    ret_data = [data[0]]
    z_data = data[0]
    i = 1
    while i < len(data):
        temp_data = k * data[i] + (1-k) * z_data
        ret_data.append(temp_data)
        z_data = temp_data
        i += 1
    
    return np.array(ret_data)

def linear_func(x, a, b):
    return a*x + b

def time_mod(times):
    ret_list = [times[0]]
    time_add = 0
    time_overflow = 2**32 / 1000000
    i = 1
    while i < len(times):
        if times[i] - times[i-1] < 0:
            time_add += time_overflow
        ret_list.append(times[i] + time_add)
        i += 1
    return np.array(ret_list)


def getWindSpeed(theta):
    if theta<0:
        direction = -1
    else:
        direction = 1
    
    f =  k*np.sin(theta * np.pi / 180)
    windSpeed = direction* np.sqrt(np.abs(2*f/(rho*cd*S_ball)))
    return windSpeed

def getWindDir(nSpeed, eSpeed):
    windDir = np.degrees(np.arctan2(-nSpeed, eSpeed)) + 180 + rot_Offset
    if windDir > 360:
        windDir -= 360
        
    if windDir <0:
        windDir += 360
        
    return windDir

def getNorm(x_val, y_val):
    return np.sqrt(x_val**2 + y_val**2)


class SensorData:
    def __init__(self, file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        
        time_list = []
        outer_val_list = []
        inter_val_list = []  
        outer_angle_list = []    
        inter_angle_list = []
        gps_list = []
        
        for line in lines:
            datas = line.split('\t')
            if len(datas) == 6:
                try:
                    time_list.append(float(datas[0])/1000000)
                    outer_val_list.append(float(datas[1]))                
                    inter_val_list.append(float(datas[2]))
                    outer_angle_list.append(float(datas[3]))
                    inter_angle_list.append(float(datas[4]))
                    gps_list.append(str(datas[5]))
                except:
                    print(line)
                    
            else:
                print(line)
                    
        sysTime = np.array(time_list)
        self.sysTime = time_mod(sysTime)
        self.outer_val = np.array(outer_val_list)
        self.inter_val = np.array(inter_val_list)
        self.outer_angle = np.array(outer_angle_list)
        self.inter_angle = np.array(inter_angle_list)
        self.gps_raw_messeage = np.array(gps_list)
        self.gps_analyze()
        
        i = 1
        outer_angle_dot = [0]
        inter_angle_dot = [0]
        outer_f = [0]
        inter_f = [0]
        outer_wind = [0]
        inter_wind = [0]
        outer_windSpeed = [0]
        inter_windSpeed = [0]
        
        while i < len(self.sysTime):
            temp_outer_angle_dot = (self.outer_angle[i] - self.outer_angle[i-1])/(self.sysTime[i]-self.sysTime[i-1])
            temp_inter_angle_dot = (self.inter_angle[i] - self.inter_angle[i-1])/(self.sysTime[i]-self.sysTime[i-1])
            outer_angle_dot.append(temp_outer_angle_dot)
            inter_angle_dot.append(temp_inter_angle_dot)
            outer_windSpeed.append(getWindSpeed(self.outer_angle[i]))
            inter_windSpeed.append(getWindSpeed(self.inter_angle[i]))
            i += 1
        self.outer_angle_dot = np.array(outer_angle_dot)
        self.inter_angle_dot = np.array(inter_angle_dot)
        self.outer_windSpeed = np.array(outer_windSpeed)
        self.inter_windSpeed = np.array(inter_windSpeed)
        self.norm_windSpeed = getNorm(self.outer_windSpeed, self.inter_windSpeed)
        i = 0
        windDir_list = []
        while i < len(self.sysTime):
            temp_windDir = getWindDir(self.outer_windSpeed[i], self.inter_windSpeed[i])
            windDir_list.append(temp_windDir)
            i += 1
        self.windDir = np.array(windDir_list)

    def gps_analyze(self):
        lon_list = []
        lat_list = []
        datetime_list = []
        
        i = 0
        
        while i < len(self.gps_raw_messeage):
            datas = self.gps_raw_messeage[i].split(';;')
            if len(datas) == 4:
                try:
                    temp_location_messeage = datas[0]
                    temp_date_messeage = datas[1]
                    temp_time_messeage = datas[2]
                    temp_location_messeage = temp_location_messeage.replace("Location:lat:","")
                    temp_loc = temp_location_messeage.split("lon:")
                    
                    
                    temp_date_messeage = temp_date_messeage.replace("Date:","")
                    temp_date = temp_date_messeage.split("/")
    
                    temp_time_messeage = temp_time_messeage.replace("Time:","")
                    temp_time = temp_time_messeage.split(":")
    
                    """temp_datatime = datetime.datetime(int(temp_date[0]), int(temp_date[1]), int(temp_date[2]), \
                                                      int(temp_time[0]), int(temp_time[1]), int(temp_time[2]), int(temp_time[3]))
                    """
                    
                    temp_datatime = 3600*(int(temp_time[0])+9) + 60*int(temp_time[1]) + int(temp_time[2]) + float(temp_time[3])/1000
                    lat_list.append(float(temp_loc[0]))
                    lon_list.append(float(temp_loc[1]))
                    datetime_list.append([self.sysTime[i], temp_datatime])
                
                except Exception as e:
                    #print(e)
                    break
            i += 1

        datetime_list = np.array(datetime_list)
        time_opt, time_cov = curve_fit(linear_func, datetime_list[:,0], datetime_list[:,1])
        #print(time_opt)
        self.lon = lon_list
        self.lat = lat_list
        self.datetime = np.array(datetime_list)
        self.time = linear_func(self.sysTime, time_opt[0], time_opt[1])
        

def getWindSpeedVector(windSpeed, windDir):
    n_speed = windSpeed * np.cos(np.pi*windDir/180)
    e_speed = windSpeed * np.sin(np.pi*windDir/180)
    return n_speed, e_speed
        
  
class ReferenceData:
    def __init__(self, file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        
        time_list = []
        error_list = []
        windSpeed_list = []    
        windDirList = []
        nSpeedList = []
        eSpeedList = []
        
        for line in lines:
            datas = line.split(',')
            if len(datas) == 4:
                try:
                    temp_datatime = datas[0]
                    temp_datatime = temp_datatime.split(" ")
                    temp_date = temp_datatime[0]
                    temp_time = temp_datatime[1]
                    temp_time = temp_time.split(":")
                    temp_time = 3600*int(temp_time[0]) + 60*int(temp_time[1]) + float(temp_time[2])
                    
                    time_list.append(temp_time)
                    error_list.append(datas[1])                
                    windSpeed_list.append(float(datas[2]))
                    windDirList.append(float(datas[3]))
                except:
                    print(line)
                    
            else:
                print(line)
        self.time = time_list
        self.error = error_list
        self.windSpeed = windSpeed_list
        self.windDir = windDirList
        i = 0
        while i < len(self.time):
            temp_nSpeed, temp_eSpeed = getWindSpeedVector(self.windSpeed[i], self.windDir[i])
            nSpeedList.append(temp_nSpeed)
            eSpeedList.append(temp_eSpeed)
            
            i += 1
            
        self.nSpeed = np.array(nSpeedList)
        self.eSpeed = np.array(eSpeedList)
        
        

sensorFile = "20250114\LOG00016.TXT"
refFile = "20250114\WS400am.txt"

sensor = SensorData(sensorFile)
reference = ReferenceData(refFile)

"""
plt.plot(sensor.time, lowpass(np.sin(np.radians(sensor.windDir)), 1), "b")
plt.plot(reference.time, np.sin(np.radians(reference.windDir)), "r--", alpha = 0.5)
plt.legend(["BallSensor", "Reference"])
plt.xlabel("JST time[sec]")
plt.ylabel("Sin ( WindDir ) [-]")
plt.title("Lowpass Gain 1")
plt.xlim([34325, 38850])
#plt.plot(reference.time, reference.nSpeed, "r--", alpha = 0.5)
#plt.plot(reference.time, reference.eSpeed, "g--", alpha = 0.5)
"""

plt.plot(sensor.time, sensor.norm_windSpeed, "b")
plt.plot(reference.time, reference.windSpeed, "r--", alpha = 0.5)
plt.legend(["BallSensor", "Reference"])
plt.xlim([34325, 38850])
plt.title("Cd = 0.55")
plt.xlabel("JST time[sec]")
plt.ylabel("WindSpeed[m/s]")

