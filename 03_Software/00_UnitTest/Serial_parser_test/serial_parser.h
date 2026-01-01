#ifndef SERIAL_PARSER_H_
#define SERIAL_PARSER_H_

#include <TWELITE>

#define MAX_BUF_SIZE 20


//https://rakko.tools/tools/74/ 

const uint8_t mode_header[5] = {0x23,0x4d,0x4f,0x44,0x45}; // #MODE
const uint8_t val_header[4] = {0x23,0x56,0x41,0x4c}; // #VAL
const uint8_t b_crlf[2] = {0x0d, 0x0a}; // \r\n
const uint8_t val_dilimiter = 0x3b; // ;

enum MES_MODE{
  MODE_HEADER,
  MODE,
  CRLF
};

enum MES_VAL{
  VAL_HEADER,
  CLS_ID,
  CRLF1,
  VAL,
  CRLF2
};



bool parseMode(uint8_t b_data, uint8_t *mode);
bool parseVal(uint8_t b_data, uint8_t *cls_id, float*val);

#endif