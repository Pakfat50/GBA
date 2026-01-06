#include <TWELITE>
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