#include <TWELITE>
#include "set_param_mode_task.h"
#include "serial_parser.h"
#include "gba_mode.h"
#include "gba_param.h"

GBA_MODE setParamModeTask(GBA_PARAM *gba_param){
    bool check_mode = false;
    uint8_t b_mode;
    uint8_t b_data;
    GBA_MODE ret_mode = SET_PARAM;
    GBA_PARAM l_gba_param;
    uint16_t param_num = 0;
    float val = 0.0;

    while(Serial.available()) {
        b_data = Serial.read();
        Serial.write(b_data);

        switch (parseVal(b_data, &param_num, &val)){
            case GET_VALUE:
                Serial.print("PARAM NUM:");
                Serial.println(param_num);
                Serial.print("VALUE:");
                Serial.println(val);

                if (set_gba_param(param_num, val) == true){
                    Serial.println("Param Write Success\n");
                    if(get_gba_param(&l_gba_param) == true){
                        *gba_param = l_gba_param;
                    }
                }else{
                    Serial.println("Param Write Error\n");
                }
                break;
            
            case ERR_INVALID_ID:
                Serial.println("PARAM_NUM inputs is invalid\n");
                break;

            case ERR_INVALID_VAL:
                Serial.println("Val inputs is invalid\n");
                break;    

            case ERR_OVER_RANGE:
                Serial.print("PARAM_NUM inputs overrange (Max size is ");
                Serial.print(PARAM_NUM_MAX);
                Serial.println(")\n");            
                break;

            default:
                break;
        }
        
        if(parseMode(b_data, &b_mode) == GET_VALUE){
            check_mode = true;
        }
    }

    if(check_mode == true){
        ret_mode = mode_checker(b_mode, SET_PARAM);
    }

    return ret_mode;

}