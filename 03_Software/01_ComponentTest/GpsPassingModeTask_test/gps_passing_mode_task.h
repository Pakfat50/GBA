#ifndef GPS_PASSING_MODE_TASK_H_
#define GPS_PASSING_MODE_TASK_H_

#include <TWELITE>
#include "gba_mode.h"

#define USE_ALT_PIN 
#define GPS_SERIAL_BAUDRATE 9600
#define GPS_SERIAL_TX_BUF_SIZE 64
#define GPS_SERIAL_RX_BUF_SIZE 192


void gpsPassingModeInit(mwx::serial_jen* gps_serial);
GBA_MODE gpsPassingModeTask(mwx::serial_jen* gps_serial, mwx::serial_jen* debug_serial);


#endif