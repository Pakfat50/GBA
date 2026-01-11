#ifndef __AHT20_H__
#define __AHT20_H__
#include <TWELITE>

void aht20_begin();
bool aht20_startSensor(void);
bool aht20_getSensor(float *h, float *t);
bool aht20_getTemperature(float *t);
bool aht20_getHumidity(float *h);

#endif