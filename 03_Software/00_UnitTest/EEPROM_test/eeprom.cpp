#include <TWELITE>
#include "eeprom.h"


bool eeprom_float_wite(uint16_t param_num, float val){
    uint16_t addr = EEPROM_ADDR_BASE + param_num*4;
    U1_R4 u1_r4;

    if (addr<EEPROM_ADDR_END)
    {
        u1_r4.R4 = val;
        for(uint16_t i=0; i<4; i++){
            EEPROM.update(addr+i, u1_r4.U1[i]);
        }
        return true;
    }
    else{
        return false;
    }
}

bool eeprom_float_read(uint16_t param_num, float *val){
    uint16_t addr = EEPROM_ADDR_BASE + param_num*4;
    if (addr<EEPROM_ADDR_END)
    {
        U1_R4 u1_r4;
        for(uint16_t i=0; i<4; i++){
            u1_r4.U1[i] = EEPROM.read(addr+i);
        }
        *val = u1_r4.R4;
        return true;
    }
    else{
        return false;
    }
}