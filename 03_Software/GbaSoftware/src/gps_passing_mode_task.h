#ifndef GPS_PASSING_MODE_TASK_H_
#define GPS_PASSING_MODE_TASK_H_

#include <TWELITE>
#include "gba_mode.h"

#define GPS_SERIAL Serial1
#define DEBUG_SERIAL Serial

GBA_MODE gpsPassingModeTask(void);

#endif