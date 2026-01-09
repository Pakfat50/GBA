#include <TWELITE>
#include "gps_passing_mode_task.h"
#include "gba_mode.h"
#include "casic_parser.h"

GBA_MODE mode = GPS_PASSING;

void setup() {
    gpsSerialInit(&Serial1);
}

void loop() {
    
    GBA_MODE next_mode = gpsPassingModeTask();

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

