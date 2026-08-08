# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:44:45 2026

@author: hirar
"""

import ctypes
import struct
import glob

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
        if len(byte_data) > 10:
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
        ('err_angle0',  ctypes.c_ubyte),
        ('err_angle1', ctypes.c_ubyte)
    ]
    def __init__(self):
        super(ANGLE_DATA, self).__init__(
            systime = 0,
            angle0 = 0,
            angle1 = 0,
            angle0_raw = 0,
            angle1_raw = 0,
            angle0_average = 0,
            angle1_average = 0,
            err_angle0 = False,
            err_angle1 = False
        )
        self.unit = {
            'systime' : "ms",
            'angle0': "deg",
            'angle1': "deg",
            'angle0_raw': "deg",
            'angle1_raw': "deg",
            'angle0_average': "deg",
            'angle1_average': "deg",
            'err_angle0' : "-",
            'err_angle1' : "-"
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
        ('pitch_raw', ctypes.c_float),
        ('err_ax',  ctypes.c_ubyte),
        ('err_ay', ctypes.c_ubyte),       
        ('err_az',  ctypes.c_ubyte),
        ('err_roll', ctypes.c_ubyte),       
        ('err_pitch',  ctypes.c_ubyte)
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
            pitch_raw = 0,
            err_ax = False,
            err_ay = False,
            err_az = False,
            err_roll = False,
            err_pitch = False
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
            'pitch_raw' : "deg",
            'err_ax' : "-",
            'err_ay' : "-",
            'err_az' : "-",
            'err_roll' : "-",
            'err_pitch' : "-"
        }


class ENV_DATA(ctypes.BigEndianStructure): 
    _pack_ = 1
    _fields_ = [
        ('systime',  ctypes.c_ulong),
        ('temperature',  ctypes.c_float),
        ('temperature_bmp280', ctypes.c_float),
        ('humidity',  ctypes.c_float),
        ('pressure', ctypes.c_float),
        ('err_humidity', ctypes.c_ubyte),       
        ('err_temperature',  ctypes.c_ubyte),
        ('err_temperature_bmp280', ctypes.c_ubyte),   
        ('err_pressure', ctypes.c_ubyte)

    ]
    def __init__(self):
        super(ENV_DATA, self).__init__(
            systime = 0,
            temperature = 0,
            temperature_bmp280 = 0,
            humidity = 0,
            pressure = 0,
            err_humidity = False,
            err_temperature = False,
            err_temperature_bmp280 = False,
            err_pressure = False
        )
        self.unit = {
            'systime' : "ms",
            'temperature': "degC",
            'temperature_bmp280': "degC",
            'humidity': "%%",
            'pressure': "hPa",
            'err_humidity' : "-",
            'err_temperature' : "-",
            'err_temperature_bmp280' : "-",
            'err_pressure' : "-"
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


class NAV_PV_DATA(ctypes.BigEndianStructure): 
    _pack_ = 1
    _fields_ = [
        ('runTime',  ctypes.c_ulong),
        ('posValid',  ctypes.c_ubyte),
        ('velValid', ctypes.c_ubyte),
        ('system',  ctypes.c_ubyte),
        ('numSV', ctypes.c_ubyte),
        ('numSVGPS',  ctypes.c_ubyte),
        ('numSVBDS', ctypes.c_ubyte),
        ('numSVGLN',  ctypes.c_ubyte),
        ('res',  ctypes.c_ubyte),
        ('pDop', ctypes.c_float),
        ('lon',  ctypes.c_double),
        ('lat', ctypes.c_double),
        ('height',  ctypes.c_float),
        ('sepGeoid', ctypes.c_float),
        ('hAcc',  ctypes.c_float),
        ('vAcc',  ctypes.c_float),
        ('velN', ctypes.c_float),
        ('velE',  ctypes.c_float),
        ('velU', ctypes.c_float),
        ('speed3D',  ctypes.c_float),
        ('speed2D', ctypes.c_float),
        ('heading',  ctypes.c_float),
        ('sAcc',  ctypes.c_float),
        ('cAcc', ctypes.c_float)  
    ]
    def __init__(self):
        super(NAV_PV_DATA, self).__init__(
            runTime = 0,
            posValid = 0,
            velValid = 0,
            system = 0,
            numSV = 0,
            numSVGPS = 0,
            numSVBDS = 0,
            numSVGLN = 0,
            res = 0,
            pDop = 0,
            lon = 0,
            lat = 0,
            height = 0,
            sepGeoid = 0,
            hAcc = 0,
            vAcc = 0,
            velN = 0,
            velE = 0,
            velU = 0,
            speed3D = 0,
            speed2D = 0,
            heading = 0,
            sAcc = 0,
            cAcc = 0
        )
        self.unit = {
            'runTime' : "ms",
            'posValid': "-",
            'velValid': "-",
            'system': "-",
            'numSV': "-",
            'numSVGPS': "-",
            'numSVBDS': "-",
            'numSVGLN' : "ms",
            'res': "-",
            'pDop': "-",
            'lon': "deg",
            'lat': "deg",
            'height': "m",
            'sepGeoid': "m",
            'hAcc' : "m^2",
            'vAcc': "m^2",
            'velN': "m/s",
            'velE': "m/s",
            'velU': "m/s",
            'speed3D': "m/s",
            'speed2D': "m/s",
            'heading' : "deg",
            'sAcc': "(m/s)^2",
            'cAcc': "deg^2"
        }


class NAV_TIMEUTC_DATA(ctypes.BigEndianStructure): 
    _pack_ = 1
    _fields_ = [
        ('runTime',  ctypes.c_ulong),
        ('tAcc',  ctypes.c_float),
        ('msErr', ctypes.c_float),
        ('ms',  ctypes.c_ushort),
        ('year', ctypes.c_ushort),
        ('month',  ctypes.c_ubyte),
        ('day', ctypes.c_ubyte),
        ('hour',  ctypes.c_ubyte),
        ('min',  ctypes.c_ubyte),
        ('sec', ctypes.c_ubyte),
        ('valid',  ctypes.c_ubyte),
        ('timeSrc', ctypes.c_ubyte),
        ('dateValid',  ctypes.c_ubyte)
    ]
    def __init__(self):
        super(NAV_TIMEUTC_DATA, self).__init__(
            runTime = 0,
            tAcc = 0,
            msErr = 0,
            ms = 0,
            year = 0,
            month = 0,
            day = 0,
            hour = 0,
            min = 0,
            sec = 0,
            valid = 0,
            timeSrc = 0,
            dateValid = 0
        )
        self.unit = {
            'runTime' : "ms",
            'tAcc': "s^2",
            'msErr': "ms",
            'ms': "ms",
            'year': "year",
            'month': "month",
            'day': "day",
            'hour' : "hour",
            'min': "min",
            'sec': "s",
            'valid': "-",
            'timeSrc': "-",
            'dateValid': "-"
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

def decodeTXT(fileName):
    f = open(fileName, "rb")
    lines = f.read()
    f.close()

    anlgeData = SerialData(b'\x47', b'\x01', ANGLE_DATA())
    imuData = SerialData(b'\x47', b'\x02', IMU_DATA())
    envData = SerialData(b'\x47', b'\x03', ENV_DATA())
    windData = SerialData(b'\x47', b'\x04', WIND_DATA())
    statusData = SerialData(b'\x47', b'\x05', STATUS_DATA())
    navPvData = SerialData(b'\x01', b'\x03', NAV_PV_DATA())
    navTimeutcData = SerialData(b'\x01', b'\x10', NAV_TIMEUTC_DATA())

    dataPackets = lines.split('##GB'.encode())

    angleDataStrList = [anlgeData.getHeader()]
    imuDataStrList = [imuData.getHeader()]
    envDataStrList = [envData.getHeader()]
    windDataStrList = [windData.getHeader()]
    statusDataStrList = [statusData.getHeader()]
    navPvDataStrList = [navPvData.getHeader()]
    navTimeutcDataStrList = [navTimeutcData.getHeader()]

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
        navPvData.setReceiveDataStatus = False
        navTimeutcData.setReceiveDataStatus = False
        
        anlgeData.setReceiveData(dataPackets[i])
        imuData.setReceiveData(dataPackets[i])
        envData.setReceiveData(dataPackets[i])
        windData.setReceiveData(dataPackets[i])
        statusData.setReceiveData(dataPackets[i])
        navPvData.setReceiveData(dataPackets[i])
        navTimeutcData.setReceiveData(dataPackets[i])    
        
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

        if navPvData.setReceiveDataStatus == True:
            navPvDataStrList.append(navPvData.getStrData(True))
            detect = True
        
        if navTimeutcData.setReceiveDataStatus == True:
            navTimeutcDataStrList.append(navTimeutcData.getStrData(True))
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
    writeStrData(fileName, "GPS_NAV_PV", navPvDataStrList)
    writeStrData(fileName, "GPS_NAV_TIMEUTC", navTimeutcDataStrList)


if __name__ == "__main__":
    fileList = glob.glob("*.TXT")
    i = 0
    while i < len(fileList):
        decodeTXT(fileList[i])
        i += 1
    