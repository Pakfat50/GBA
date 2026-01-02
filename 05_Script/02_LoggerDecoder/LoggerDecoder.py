# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:44:45 2026

@author: hirar
"""

import ctypes
import struct


def convertBytesToStructure(struct, byte):
    if (ctypes.sizeof(struct) == len(byte)):
        ctypes.memmove(ctypes.addressof(struct), byte, ctypes.sizeof(struct))

def calCheckSum(byte_array):
    ck_sum = 0
    for byte in byte_array:
        ck_sum += int(byte)
    return ck_sum


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


angle = ANGLE() 

f = open("LOG00002.TXT", "rb")
lines = f.read()
f.close()

b_datas = lines.split(b'\x3b\x3b\x47\x42')
b_data = b_datas[1]

b_cksum = b_data[-4:]
b_clsid = b_data[0]
b_sub_id = b_data[1]
b_len = b_data[2:4]
b_payload = b_data[4:-4]
b_data4cksum = b_data[:-4]

length = int.from_bytes(b_len, 'little')
cksum = int.from_bytes(b_cksum, 'little')
calc_cksum = calCheckSum(b_data4cksum)

convertBytesToStructure(angle, b_payload)

print(angle.time)
print(angle.angle1)
print(angle.angle2)
print("0x{0:02x}".format(angle.id))

print("0x{0:02x}".format(b_clsid))
print("0x{0:02x}".format(b_sub_id))
print(length)
print(cksum)
print(calc_cksum)