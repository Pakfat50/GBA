#include <TWELITE>
#include "gba_mode.h"

GBA_MODE mode_checker(uint8_t b_mode, GBA_MODE now_mode){
    GBA_MODE ret_mode = now_mode;
    switch (b_mode){
        case ASCII_N:
            ret_mode = NORMAL;
            break;

        case ASCII_S:
            ret_mode = SET_PARAM;
            break;
 
        case ASCII_V:
            ret_mode = VIEW_PARAM;
            break;

        case ASCII_G:
            ret_mode = GPS_PASSING;
            break;

        default:
            break;
    }

    return ret_mode;
}