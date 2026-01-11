#include <TWELITE>
#include <math.h>
#include <stdio.h>
#include "calc_wind_speed.h"

float calcWindSpeed(float angle, float coff_angle){
    float rad_angle = angle * (PI/180.0f);
    float wind_speed = sqrt(tan( fabsf(rad_angle) )/coff_angle);

    if(angle < 0){
        return -wind_speed;
    }
    else{
        return wind_speed;
    }
}