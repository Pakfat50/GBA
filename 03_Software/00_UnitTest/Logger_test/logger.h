#ifndef LOGGER_H_
#define LOGGER_H_

#include <TWELITE>

const uint8_t header[2] = {0x47, 0x42}; // GB
const uint8_t footer[2] = {0x3b, 0x3b}; // ;;


typedef union {
  uint8_t U1[2];
  uint16_t U2;
}U1_U2;

typedef union {
  uint8_t U1[4];
  uint32_t U4;
}U1_U4;

void send_data(uint8_t clsid, uint8_t sub_id, uint16_t len, uint8_t payload[], mwx::serial_jen* ser);

#endif