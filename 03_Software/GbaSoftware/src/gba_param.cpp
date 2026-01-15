#include <TWELITE>
#include "gba_param.h"
#include "eeprom.h"
#include "utility.h"

bool get_gba_param(GBA_PARAM *gba_param){
    GBA_PARAM l_gba_param;
    bool read_success = true;

    if (eeprom_float_read(0, &l_gba_param.angleOffset0) == false){
        l_gba_param.angleOffset0 = ANGLE_OFFSET_DEFAULT;
        read_success = false;
    }
    if (eeprom_float_read(1, &l_gba_param.angleOffset1) == false){
        l_gba_param.angleOffset1 = ANGLE_OFFSET_DEFAULT;
        read_success = false;        
    }
    if (eeprom_float_read(2, &l_gba_param.rollOffset) == false){
        l_gba_param.rollOffset = ROLL_OFFSET_DEFAULT;
        read_success = false; 
    }
    if (eeprom_float_read(3, &l_gba_param.pitchOffset) == false){
        l_gba_param.pitchOffset = PITCH_OFFSET_DEFAULT;
        read_success = false;         
    }
    if (eeprom_float_read(4, &l_gba_param.coffAngle0) == false){
        l_gba_param.coffAngle0 = COFF_ANGLE_DEFAULT;
        read_success = false;         
    }
    if (eeprom_float_read(5, &l_gba_param.coffAngle1) == false){
        l_gba_param.coffAngle1 = COFF_ANGLE_DEFAULT;
        read_success = false;         
    }


    if (range_check(l_gba_param.angleOffset0, &l_gba_param.angleOffset0, ANGLE_OFFSET_MAX, ANGLE_OFFSET_MIN, ANGLE_OFFSET_DEFAULT) == true){
        read_success = false;
    }
    if (range_check(l_gba_param.angleOffset1, &l_gba_param.angleOffset1, ANGLE_OFFSET_MAX, ANGLE_OFFSET_MIN, ANGLE_OFFSET_DEFAULT) == true){
        read_success = false;
    }    
    if (range_check(l_gba_param.rollOffset, &l_gba_param.rollOffset, ROLL_OFFSET_MAX, ROLL_OFFSET_MIN, ROLL_OFFSET_DEFAULT) == true){
        read_success = false;
    }    
    if (range_check(l_gba_param.pitchOffset, &l_gba_param.pitchOffset, PITCH_OFFSET_MAX, PITCH_OFFSET_MIN, PITCH_OFFSET_DEFAULT) == true){
        read_success = false;
    }    
    if (range_check(l_gba_param.coffAngle0, &l_gba_param.coffAngle0, COFF_ANGLE_MAX, COFF_ANGLE_MIN, COFF_ANGLE_DEFAULT) == true){
        read_success = false;
    }    
    if (range_check(l_gba_param.coffAngle1, &l_gba_param.coffAngle1, COFF_ANGLE_MAX, COFF_ANGLE_MIN, COFF_ANGLE_DEFAULT) == true){
        read_success = false;
    }    

    *gba_param = l_gba_param;

    return read_success;
}

bool set_gba_param(uint16_t param_num, float val){
    bool write_sucsess = false;
    float temp_val;

    switch (param_num)
    {
        case 0:
            if (range_check(val, &temp_val, ANGLE_OFFSET_MAX, ANGLE_OFFSET_MIN, ANGLE_OFFSET_DEFAULT) == false){
                if (eeprom_float_wite(0, val) == true){
                    write_sucsess = true;
                }
            }
            break;

        case 1:
            if (range_check(val, &temp_val, ANGLE_OFFSET_MAX, ANGLE_OFFSET_MIN, ANGLE_OFFSET_DEFAULT) == false){
                if (eeprom_float_wite(1, val) == true){
                    write_sucsess = true;
                }
            }            
            break;
            
        case 2:
            if (range_check(val, &temp_val, ROLL_OFFSET_MAX, ROLL_OFFSET_MIN, ROLL_OFFSET_DEFAULT) == false){
                if (eeprom_float_wite(2, val) == true){
                    write_sucsess = true;
                }
            }            
            break;
            
        case 3:
            if (range_check(val, &temp_val, PITCH_OFFSET_MAX, PITCH_OFFSET_MIN, PITCH_OFFSET_DEFAULT) == false){
                if (eeprom_float_wite(3, val) == true){
                    write_sucsess = true;
                }
            }            
            break;
            
        case 4:
            if (range_check(val, &temp_val, COFF_ANGLE_MAX, COFF_ANGLE_MIN, COFF_ANGLE_DEFAULT) == false){
                if (eeprom_float_wite(4, val) == true){
                    write_sucsess = true;
                }
            }            
            break;
            
        case 5:
            if (range_check(val, &temp_val, COFF_ANGLE_MAX, COFF_ANGLE_MIN, COFF_ANGLE_DEFAULT) == false){
                if (eeprom_float_wite(5, val) == true){
                    write_sucsess = true;
                }
            }            
            break;            

        default:
            break;
    }

    return write_sucsess;

}