#include <TWELITE>
#include "gps_passing_mode_task.h"
#include "serial_parser.h"
#include "gba_mode.h"


GBA_MODE gpsPassingModeTask(void){
    bool check_mode = false;
    uint8_t b_mode;
    GBA_MODE ret_mode = GPS_PASSING;

    while(Serial1.available()) {
        auto c = Serial1.read();
        Serial << char_t(c);
    }

    while(Serial.available()) {
        auto c = Serial.read();
        Serial1 << char_t(c);

        if(parseMode(c, &b_mode) == GET_VALUE){
            check_mode = true;
        }
    }

    if(check_mode == true){
        ret_mode = mode_checker(b_mode, GPS_PASSING);
    }

    return ret_mode;
}
