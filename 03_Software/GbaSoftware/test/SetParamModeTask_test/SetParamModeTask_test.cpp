#include <TWELITE>
#include "set_param_mode_task.h"
#include "gba_mode.h"
#include "gba_param.h"

GBA_MODE mode = SET_PARAM;
GBA_PARAM gba_param;

static void printParam(const char param_name[], float param_val);

void setup() {
    get_gba_param(&gba_param);
}

void loop() {

    GBA_MODE next_mode = setParamModeTask(&gba_param);

    if(mode != next_mode){
        printParam("angleOffset0", gba_param.angleOffset0);
        printParam("angleOffset1", gba_param.angleOffset1);
        printParam("rollOffset", gba_param.rollOffset);
        printParam("pitchOffset", gba_param.pitchOffset);
        printParam("coffAngle0", gba_param.coffAngle0);
        printParam("coffAngle1", gba_param.coffAngle1);
        Serial.println();        


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

static void printParam(const char param_name[], float param_val){
    Serial.print(param_name);
    Serial.print(":\t");
    Serial.println(param_val);
}