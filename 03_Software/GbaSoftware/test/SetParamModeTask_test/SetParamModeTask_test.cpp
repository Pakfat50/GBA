#include <TWELITE>
#include "set_param_mode_task.h"
#include "gba_mode.h"

GBA_MODE mode = SET_PARAM;

void setup() {

}

void loop() {

    GBA_MODE next_mode = setParamModeTask();

    if(mode != next_mode){
        switch (next_mode)
        {
        case NORMAL:
            Serial.println("\n\nNext mode is NORMAL\n\n");
            break;

        case SET_PARAM:
            Serial.println("\n\nNext mode is SET_PARAM\n\n");
            break;

        case VIEW_PARAM:
            Serial.println("\n\nNext mode is VIEW_PARAM\n\n");
            break;

        case GPS_PASSING:
            Serial.println("\n\nNext mode is GPS_PASSING\n\n");
            break;
        
        default:
            break;
        }
    }

}

