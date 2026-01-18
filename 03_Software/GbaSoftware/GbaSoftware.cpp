#include <TWELITE>
#include "normal_mode_task.h"
#include "gps_passing_mode_task.h"
#include "set_param_mode_task.h"
#include "view_param_mode_task.h"
#include "gba_mode.h"
#include "gba_param.h"

GBA_PARAM gba_param;
GBA_MODE mode = NORMAL;

void setup() {
    normalModeInit();
    get_gba_param(&gba_param);
}

void loop() {
    GBA_MODE z_mode = mode;

    switch (mode)
    {
        case NORMAL:
            mode = normalModeTask(gba_param);
            break;

        case SET_PARAM:
            mode = setParamModeTask(&gba_param);
            break;

        case VIEW_PARAM:
            mode = viewParamModeTask();
            break;

        case GPS_PASSING:
            mode = gpsPassingModeTask();
            break;

        default:
            break;
    }
    
    if(mode != z_mode){
        switch (mode)
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

