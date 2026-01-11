#include <TWELITE>
#include "normal_mode_task.h"
#include "gba_mode.h"
#include "gba_param.h"

GBA_MODE mode = NORMAL;
GBA_PARAM gba_param;

void setup() {
    normalModeInit();
    gba_param.angleOffset0 = 0.0;
    gba_param.angleOffset1 = 0.0;
    gba_param.rollOffset = 0.0;
    gba_param.pitchOffset = 0.0;
    gba_param.coffAngle0 = 0.04174;
    gba_param.coffAngle1 = 0.04174;    
}

void loop() {
    GBA_MODE next_mode = normalModeTask(gba_param);

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

