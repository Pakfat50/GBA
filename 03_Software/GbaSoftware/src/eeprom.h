#ifndef EEPROM_H_
#define EEPROM_H_

#include <TWELITE>

#define EEPROM_ADDR_BASE 1024
#define EEPROM_ADDR_END 0xEF0 //0xEFF - 16byte

bool eeprom_float_wite(uint16_t param_num, float val);
bool eeprom_float_read(uint16_t param_num, float *val);

#endif