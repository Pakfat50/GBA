#ifndef GBA_MODE_
#define GBA_MODE_

#include <TWELITE>

#define ASCII_N 0x4e
#define ASCII_S 0x53
#define ASCII_V 0x56
#define ASCII_G 0x47

enum GBA_MODE{
  NORMAL,
  SET_PARAM,
  VIEW_PARAM,
  GPS_PASSING
};

GBA_MODE mode_checker(uint8_t b_mode, GBA_MODE now_mode);


#endif