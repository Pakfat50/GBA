#ifndef LOGGER_H_
#define LOGGER_H_

#include <TWELITE>

const uint8_t header[4] = {0x23, 0x23, 0x47, 0x42}; // ##GB

void send_data(uint8_t clsid, uint8_t sub_id, uint16_t len, uint8_t payload[], mwx::serial_jen* ser);

#endif