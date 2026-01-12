#include <TWELITE>
#include <math.h>
#include <stdio.h>
#include "utility.h"

void LOWPASS::init(float k, float x0){
    _k = k;
    _zx = x0;
}

float LOWPASS::get(float x){
    float x_lowp = _k*x + (1-_k)*_zx;
    _zx = x_lowp;
    return x_lowp;
}

void MOVING_AVERAGE::init(uint32_t average_num, float x0){
    _average_num = average_num;
    _num = 0;
    _coff = 1.0 / float(average_num);
    _ret_val = x0;
    _zx = 0;
}

float MOVING_AVERAGE::get(float x){
    if(_num >= _average_num){
        _ret_val = _zx;
        _zx = _coff*x;
        _num = 1;
    }
    else{
        _zx += _coff*x;
        _num += 1;
    }

    return _ret_val;
}

void MIN_MAX::init(float x0){
    _x_max = x0;
    _x_min = x0;
}

void MIN_MAX::reset(float x0){
    init(x0);
}

void MIN_MAX::set(float x){
    if(x > _x_max){
        _x_max = x;
    }
    if(x < _x_min){
        _x_min = x;
    }
}

float MIN_MAX::getMin(void){
    return _x_min;
}

float MIN_MAX::getMax(void){
    return _x_max;
}

float MIN_MAX::getMaxNorm(void){
    if(fabsf(_x_max) > fabsf(_x_min)){
        return _x_max;
    }
    else{
        return _x_min;
    }
}

uint32_t micros(void){
    uint32_t milli = millis();
    uint32_t micro = u32AHI_TickTimerRead()/16 + milli*1000;
    
    return micro;
}

bool range_check(float *val, float range_max, float range_min, float val_default){
    bool range_err = false;

    if((std::isfinite(*val) == false) || (*val > range_max) || (*val < range_min)){
        *val = val_default;
        range_err = true;
    }

    return range_err;
}