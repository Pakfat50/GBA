# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:44:45 2026

@author: hirar
"""

import ctypes
import struct


class SerialData:
    def __init__(self, clsId, subId, CtypeClass, structName = ""):
        self.structName = structName
        self.t = CtypeClass
        self.clsId = clsId
        self.subId = subId
        self.length = ctypes.sizeof(self.t)
        self.cksumErr = False
        self.setReceiveDataStatus = False

    def setReceiveData(self, byte_data):
         clsId = byte_data[0].to_bytes(1,'big')
         subId = byte_data[1].to_bytes(1,'big')
         length = int.from_bytes(byte_data[2:4],'big')
         payload = byte_data[4:-4]
         checkSum = int.from_bytes(byte_data[-4:],'big')
         
         if (clsId == self.clsId) and (subId == self.subId) and (length == self.length):
            calcCheckSum = calCheckSum(clsId, subId, length, payload)
            
            if(calcCheckSum == checkSum):
                self.cksumErr = False
                try:
                    convertBytesToStructure(self.t, payload)
                    self.setReceiveDataStatus = True #受信成功
                except:
                    print("0x{0:02x}".format(self.clsId), "0x{0:02x}".format(self.subId))
            else:
                #print('CKSum does not match at %s%s'%(self.__header_name[0],self.__header_name[1]))
                self.cksumErr = True

    def getStrData(self, det, filMode = "Blank"):
        tempStr = ""
        for name, dtype in self.t._fields_:
            temp_data =  getattr(self.t, name)
            if (((type(temp_data) == float) or \
                (type(temp_data) == bool) or \
                (type(temp_data) == int) or \
                (type(temp_data) == str))):
                
                tempStrData = "%s"%temp_data
            else:
                tempStrData = ""
            
            if det == True:
                tempStr += tempStrData + ","
                
            else:
                if filMode == "Blank":
                    tempStr += "" + ","
                elif filMode == "Zval":
                    tempStr += tempStrData + ","
                elif filMode == "Nan":
                    tempStr += "Nan" + ","
        return tempStr

    def getHeader(self):
        tempStr = ""
        for name, dtype in self.t._fields_:
            tempUnit = self.t.unit[name]
            tempStr += "%s%s[%s],"%(name,self.structName,tempUnit)
        return tempStr   

    def getDictData(self):
        temp_list = []
        for name, dtype in self.t._fields_:
            temp_data =  getattr(self.t, name)
            if ((type(temp_data) == float) or \
                (type(temp_data) == bool) or \
                (type(temp_data) == int) or \
                (type(temp_data) == str)) :
                temp_list.append( [name, temp_data] )
            else:
                temp_list.append( [name, 0] )
        return dict(temp_list)
    

def convertBytesToStructure(struct, byte):
    if (ctypes.sizeof(struct) == len(byte)):
        ctypes.memmove(ctypes.addressof(struct), byte, ctypes.sizeof(struct))


def calCheckSum(clsId, subId, length, payload):
    b_length = length.to_bytes(2, 'big')
    
    checkSum = 0
    checkSum += int.from_bytes(clsId,'big')
    checkSum += int.from_bytes(subId,'big')
    checkSum += int(b_length[0])
    checkSum += int(b_length[1])
    
    for byte in payload:
        checkSum += int(byte)
        
    return checkSum


class ANGLE(ctypes.BigEndianStructure): 
    _pack_ = 1
    _fields_ = [
        ('time',  ctypes.c_ulong),
        ('angle1',  ctypes.c_float),
        ('angle2', ctypes.c_float),
        ('id',   ctypes.c_ubyte)    
    ]
    def __init__(self):
        super(ANGLE, self).__init__(
            time = 0,
            angle1 = 0,
            angle2 = 0,
            id = 0
        )
        self.unit = {
            'time' : "ms",
            'angle1': "deg",
            'angle2': "deg",
            "id" : ""
        }

class ANGLE_DATA(ctypes.BigEndianStructure): 
    _pack_ = 1
    _fields_ = [
        ('systime',  ctypes.c_ulong),
        ('angle0',  ctypes.c_float),
        ('angle1', ctypes.c_float),
        ('angle0_raw',  ctypes.c_float),
        ('angle1_raw', ctypes.c_float),
        ('angle0_average',  ctypes.c_float),
        ('angle1_average', ctypes.c_float),
    ]
    def __init__(self):
        super(ANGLE_DATA, self).__init__(
            systime = 0,
            angle0 = 0,
            angle1 = 0,
            angle0_raw = 0,
            angle1_raw = 0,
            angle0_average = 0,
            angle1_average = 0
        )
        self.unit = {
            'systime' : "ms",
            'angle0': "deg",
            'angle1': "deg",
            'angle0_raw': "deg",
            'angle1_raw': "deg",
            'angle0_average': "deg",
            'angle1_average': "deg"
        }

class IMU_DATA(ctypes.BigEndianStructure): 
    _pack_ = 1
    _fields_ = [
        ('systime',  ctypes.c_ulong),
        ('ax',  ctypes.c_float),
        ('ay', ctypes.c_float),
        ('az',  ctypes.c_float),
        ('gx', ctypes.c_float),
        ('gy',  ctypes.c_float),
        ('gz', ctypes.c_float),
        ('roll',  ctypes.c_float),
        ('pitch', ctypes.c_float),
        ('roll_raw',  ctypes.c_float),
        ('pitch_raw', ctypes.c_float)
    ]
    def __init__(self):
        super(IMU_DATA, self).__init__(
            systime = 0,
            ax = 0,
            ay = 0,
            az = 0,
            gx = 0,
            gy = 0,
            gz = 0,
            roll = 0,
            pitch = 0,
            roll_raw = 0,
            pitch_raw = 0
        )
        self.unit = {
            'systime' : "ms",
            'ax': "g",
            'ay': "g",
            'az': "g",
            'gx': "deg/s",
            'gy': "deg/s",
            'gz': "deg/s",
            'roll' : "deg",
            'pitch' : "deg",
            'roll_raw' : "deg",
            'pitch_raw' : "deg"
        }


class ENV_DATA(ctypes.BigEndianStructure): 
    _pack_ = 1
    _fields_ = [
        ('systime',  ctypes.c_ulong),
        ('temperature',  ctypes.c_float),
        ('temperature_bmp280', ctypes.c_float),
        ('humidity',  ctypes.c_float),
        ('pressure', ctypes.c_float)

    ]
    def __init__(self):
        super(ENV_DATA, self).__init__(
            systime = 0,
            temperature = 0,
            temperature_bmp280 = 0,
            humidity = 0,
            pressure = 0
        )
        self.unit = {
            'systime' : "ms",
            'temperature': "degC",
            'temperature_bmp280': "degC",
            'humidity': "%%",
            'pressure': "hPa"
        }


class WIND_DATA(ctypes.BigEndianStructure): 
    _pack_ = 1
    _fields_ = [
        ('systime',  ctypes.c_ulong),
        ('average_east',  ctypes.c_float),
        ('average_north', ctypes.c_float),
        ('gust_east',  ctypes.c_float),
        ('gust_north', ctypes.c_float)
    ]
    def __init__(self):
        super(WIND_DATA, self).__init__(
            systime = 0,
            average_east = 0,
            average_north = 0,
            gust_east = 0,
            gust_north = 0
        )
        self.unit = {
            'systime' : "ms",
            'average_east': "m/s",
            'average_north': "m/s",
            'gust_east': "m/s",
            'gust_north': "m/s"
        }

class STATUS_DATA(ctypes.BigEndianStructure): 
    _pack_ = 1
    _fields_ = [
        ('systime',  ctypes.c_ulong),
        ('mag_strength0',  ctypes.c_uint),
        ('mag_strength1', ctypes.c_uint),
        ('push_botton0',  ctypes.c_uint),
        ('push_botton1', ctypes.c_uint),
        ('track0',  ctypes.c_uint),
        ('track1', ctypes.c_uint)
    ]
    def __init__(self):
        super(STATUS_DATA, self).__init__(
            systime = 0,
            mag_strength0 = 0,
            mag_strength1 = 0,
            push_botton0 = 0,
            push_botton1 = 0,
            track0 = 0,
            track1 = 0
        )
        self.unit = {
            'systime' : "ms",
            'mag_strength0': "-",
            'mag_strength1': "-",
            'push_botton0': "-",
            'push_botton1': "-",
            'track0': "-",
            'track1': "-"
        }

def writeStrData(fileName, dataType, strList):
    dataName = fileName.replace(".TXT", "")
    dataName += "_"
    dataName += dataType
    dataName += ".csv"
    
    f = open(dataName, "w")
    for line in strList:
        f.write(line)
        f.write('\n')
    f.close()


fileName = "LOG00004.TXT"

f = open(fileName, "rb")
lines = f.read()
f.close()

anlgeData = SerialData(b'\x47', b'\x01', ANGLE_DATA())
imuData = SerialData(b'\x47', b'\x02', IMU_DATA())
envData = SerialData(b'\x47', b'\x03', ENV_DATA())
windData = SerialData(b'\x47', b'\x04', WIND_DATA())
statusData = SerialData(b'\x47', b'\x05', STATUS_DATA())

dataPackets = lines.split('##GB'.encode())

angleDataStrList = [anlgeData.getHeader()]
imuDataStrList = [imuData.getHeader()]
envDataStrList = [envData.getHeader()]
windDataStrList = [windData.getHeader()]
statusDataStrList = [statusData.getHeader()]

detect = False
lostPacketNum = 0
totalDataPacketNum = len(dataPackets)

i = 0
while i < len(dataPackets):
#while i < 20:
    detect = False
    anlgeData.setReceiveDataStatus = False
    imuData.setReceiveDataStatus = False
    envData.setReceiveDataStatus = False
    windData.setReceiveDataStatus = False
    statusData.setReceiveDataStatus = False
    
    anlgeData.setReceiveData(dataPackets[i])
    imuData.setReceiveData(dataPackets[i])
    envData.setReceiveData(dataPackets[i])
    windData.setReceiveData(dataPackets[i])
    statusData.setReceiveData(dataPackets[i])
    
    if anlgeData.setReceiveDataStatus == True:
        angleDataStrList.append(anlgeData.getStrData(True))
        detect = True
        
    if imuData.setReceiveDataStatus == True:
        imuDataStrList.append(imuData.getStrData(True))
        detect = True
        
    if envData.setReceiveDataStatus == True:
        envDataStrList.append(envData.getStrData(True))
        detect = True
        
    if windData.setReceiveDataStatus == True:
        windDataStrList.append(windData.getStrData(True))
        detect = True
        
    if statusData.setReceiveDataStatus == True:
        statusDataStrList.append(statusData.getStrData(True))
        detect = True
    
    if detect == False:
        lostPacketNum += 1
        
    print("処理中。{:.1f}%".format(float(i*100/totalDataPacketNum)))
    
    i += 1

print("ロストパケット数： %s \n"%lostPacketNum)

writeStrData(fileName, "ANGLE", angleDataStrList)
writeStrData(fileName, "IMU", imuDataStrList)
writeStrData(fileName, "ENV", envDataStrList)
writeStrData(fileName, "WIND", windDataStrList)
writeStrData(fileName, "STATUS", statusDataStrList)

