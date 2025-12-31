# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 17:40:31 2025

@author: hirar
"""


import serial as ser
import struct
import time


_serial = ser.Serial()
_serial.port = "COM7"
_serial.baundrate = 115200

_serial.parity = ser.PARITY_NONE
_serial.bytesize = ser.EIGHTBITS
_serial.stopbits = ser.STOPBITS_ONE
_serial.xonxoff=False
_serial.rtscts=False
_serial.dsrdtr=False

_serial.open()
_serial.flush()



while True:
    print(_serial.read_all())
    time.sleep(0.1)
