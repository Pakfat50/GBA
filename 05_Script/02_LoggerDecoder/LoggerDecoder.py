# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:44:45 2026

@author: hirar
"""

import ctypes
import struct


class SerialData:
    def __init__(self, clsId, subId, CtypeClass):
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


f = open("LOG00003.TXT", "rb")
lines = f.read()
f.close()

anlgeData = SerialData(b'\x01', b'\x02', ANGLE())

b_datas = lines.split('##GB'.encode())

anlgeData.setReceiveData(b_datas[3])
print(anlgeData.t.time)
print(anlgeData.t.angle1)
print(anlgeData.t.angle2)
print(anlgeData.t.id)

