#ifndef GBA_COMMON_H_
#define GBA_COMMON_H_

#include <TWELITE>

#define EEPROM_ADDR_BASE 1024
#define EEPROM_ADDR_END 0xEF0 //0xEFF - 16byte

typedef union {
  uint8_t U1[4];
  uint32_t U4;
}U1_U4;

typedef union {
  uint8_t U1[2];
  uint16_t U2;
}U1_U2;

typedef union {
  uint8_t U1[4];
  int32_t I4;
}U1_I4;

typedef union {
  uint8_t U1[2];
  int16_t I2;
}U1_I2;

typedef union {
  uint8_t U1;
  int8_t I1;
}U1_I1;

typedef union {
  uint8_t U1[4];
  float R4;
}U1_R4;

typedef union {
  uint8_t U1[8];
  double R8;
}U1_R8;



#endif