#ifndef NORMAL_MODE_TASK_H_
#define NORMAL_MODE_TASK_H_

#include <TWELITE>
#include "gba_mode.h"
#include "gba_param.h"
#include "mt6701.h"

#define DEBUG_NORMAL_MODE

#define MS_TO_US 1000   // [-]
#define BASE_RATE 100   // [Hz]
#define TIME_INTERVAL_MS 1000/BASE_RATE // [ms]
#define TIME_INTERVAL_US 1000000/BASE_RATE // [us]

#define TIME_WIND_AVERAGE 3000 //ms
#define NUM_WIND_AVERAGE TIME_WIND_AVERAGE*BASE_RATE // [-]
#define TRANSMIT_INT 50 // [ms]
#define TRANSMIT_ADDR 0xff 
#define TIME_PPS_MAX_WAIT 1200 // [ms]

#define DELTA_T_MAX 10000000 // 10sec

#define GBA_CLASS_ID 0x47
#define ANGLE_ID 0x01
#define IMU_ID 0x02
#define ENV_ID 0x03
#define WIND_ID 0x04
#define STATUS_ID 0x05


#pragma pack(push, 1)
typedef struct{
    uint32_t systime;
    float angle0;
    float angle1;
    float angle0_raw;
    float angle1_raw;
    float angle0_average;
    float angle1_average;
}ANGLE_DATA;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct{
    uint32_t systime;
    float ax;
    float ay;
    float az;
    float gx;
    float gy;
    float gz;
    float roll;
    float pitch;
    float roll_raw;
    float pitch_raw;
}IMU_DATA;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct{
    uint32_t systime;
    float temperature;
    float temperature_bmp280;
    float humidity;
    float pressure;
}ENV_DATA;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct{
    uint32_t systime;
    float average_east;
    float average_north;
    float gust_east;
    float gust_north;
}WIND_DATA;
#pragma pack(pop)


#pragma pack(push, 1)
typedef struct{
    uint32_t systime;
    MAG_STRENGTH mag_strength0;
    MAG_STRENGTH mag_strength1;
    PUSH_BOTTON push_botton0;
    PUSH_BOTTON push_botton1;
    TRACK track0;
    TRACK track1;
}STATUS_DATA;
#pragma pack(pop)

void normalModeInit(void);

GBA_MODE normalModeTask(GBA_PARAM gba_param);

#endif