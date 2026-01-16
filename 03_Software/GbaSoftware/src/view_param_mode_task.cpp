#include <TWELITE>
#include "view_param_mode_task.h"
#include "serial_parser.h"
#include "gba_mode.h"
#include "gba_param.h"

static void printParam(const char param_name[], float param_val);

GBA_MODE viewParamModeTask(void){
    bool check_mode = false;
    uint8_t b_mode;
    uint8_t b_data;
    GBA_MODE ret_mode = VIEW_PARAM;
    GBA_PARAM l_gba_param;

    while(Serial.available()) {
        b_data = Serial.read();
        
        if(parseMode(b_data, &b_mode) == GET_VALUE){
            check_mode = true;
        }
    }

    if(check_mode == true){
        if (b_data == ASCII_V){
            if(get_gba_param(&l_gba_param) == true){
                printParam("angleOffset0", l_gba_param.angleOffset0);
                printParam("angleOffset1", l_gba_param.angleOffset1);
                printParam("rollOffset", l_gba_param.rollOffset);
                printParam("pitchOffset", l_gba_param.pitchOffset);
                printParam("coffAngle0", l_gba_param.coffAngle0);
                printParam("coffAngle1", l_gba_param.coffAngle1);
                Serial.println();
            }else{
                Serial.println("Param read error!");
            }
        }

        ret_mode = mode_checker(b_mode, VIEW_PARAM);
    }

    return ret_mode;

}

static void printParam(const char param_name[], float param_val){
    Serial.print(param_name);
    Serial.print(":\t");
    Serial.println(param_val);
}

