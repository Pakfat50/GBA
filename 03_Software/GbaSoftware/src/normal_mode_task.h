#ifndef NORMAL_MODE_TASK_H_
#define NORMAL_MODE_TASK_H_

#include <TWELITE>
#include "gba_mode.h"
#include "gba_param.h"
#include "mt6701.h"

//#define DEBUG_NORMAL_MODE

#define MS_TO_US 1000   // [-]
#define BASE_RATE 100   // [Hz]
#define TIME_INTERVAL_MS 1000/BASE_RATE // [ms]
#define TIME_INTERVAL_US 1000000/BASE_RATE // [us]
#define DELAY_TIME_AJUST 2 // [us]

#define TIME_WIND_AVERAGE 3 //sec
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

#define ANGLE_MAX 3600.0 // 10周
#define ANGLE_MIN -3600.0 // -10周
#define ANGLE_DEFAULT 0.0

#define ACC_MAX 2.0 // G
#define ACC_MIN -2.0 //G
#define ACC_DEFAULT 0.0001 //G 零割防止のため、微小値とする

#define ROLL_MAX 361.0 // 演算誤差防止のため、+1deg
#define ROLL_MIN -361.0 // 演算誤差防止のため、-1deg
#define ROLL_DEFAULT 0.0

#define PITCH_MAX 361.0 // 演算誤差防止のため、+1deg
#define PITCH_MIN -361.0 // 演算誤差防止のため、-1deg
#define PITCH_DEFAULT 0.0

#define HUMIDITY_MAX 100.0 //%
#define HUMIDITY_MIN  0.0 //%
#define HUMIDITY_DEFAULT 50.0 //%

#define TEMPERATURE_MAX 60.0 //degC
#define TEMPERATURE_MIN -20.0 //degC
#define TEMPERATURE_DEFAULT 20.0 //degC

#define PRESSURE_MAX 1200.0 //hPa
#define PRESSURE_MIN 900.0 //hPa
#define PRESSURE_DEFAULT 1013.15 //hPa

#define ASCII_A 0x41
#define ASCII_B 0x42

#define DELIMITER "\t"

#pragma pack(push, 1)
typedef struct{
    uint32_t systime;
    float angle0;
    float angle1;
    float angle0_raw;
    float angle1_raw;
    float angle0_average;
    float angle1_average;
    bool err_angle0;
    bool err_angle1;
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
    bool err_ax;
    bool err_ay;
    bool err_az;
    bool err_roll;
    bool err_pitch;
}IMU_DATA;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct{
    uint32_t systime;
    float temperature;
    float temperature_bmp280;
    float humidity;
    float pressure;
    bool err_humidity;
    bool err_temperature;
    bool err_temperature_bmp280;
    bool err_pressure;
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