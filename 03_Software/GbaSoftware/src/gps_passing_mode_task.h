#ifndef GPS_PASSING_MODE_TASK_H_
#define GPS_PASSING_MODE_TASK_H_

#include <TWELITE>
#include "gba_mode.h"

#define USE_ALT_PIN 
#define GPS_SERIAL_BAUDRATE 9600
#define GPS_SERIAL_TX_BUF_SIZE 64
#define GPS_SERIAL_RX_BUF_SIZE 192
#define GPS_SERIAL Serial1
#define DEBUG_SERIAL Serial

void gpsPassingModeInit(void);
GBA_MODE gpsPassingModeTask(void);


#endif