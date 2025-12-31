#ifndef CASIC_PARSER_H_
#define CASIC_PARSER_H_

#include <TWELITE>

#define MAX_BUF_SIZE 120

const uint8_t mes_disable_nema[]    = {0xBA,0xCE,0x08,0x00,0x06,0x00,0xFF,0x13,0xC0,0x08,0x80,0x25,0x00,0x00,0x87,0x39,0xC6,0x08};
const uint8_t mes_enable_nema[]     = {0xBA,0xCE,0x08,0x00,0x06,0x00,0xFF,0x33,0xC0,0x08,0x80,0x25,0x00,0x00,0x87,0x59,0xC6,0x08};
const uint8_t mes_disable_pv[]      = {0xBA,0xCE,0x04,0x00,0x06,0x01,0x01,0x00,0x00,0x00,0x05,0x00,0x06,0x01}; 
const uint8_t mes_disable_timeutc[] = {0xBA,0xCE,0x04,0x00,0x06,0x01,0x01,0x10,0x00,0x00,0x05,0x10,0x06,0x01}; 
const uint8_t mes_enable_pv[]       = {0xBA,0xCE,0x04,0x00,0x06,0x01,0x01,0x03,0x01,0x00,0x05,0x03,0x07,0x01};
const uint8_t mes_enable_timeutc[]  = {0xBA,0xCE,0x04,0x00,0x06,0x01,0x01,0x10,0x01,0x00,0x05,0x10,0x07,0x01};
const char str_nema_pcsa04[] = "$PCAS04,2C*58\r\n";

#pragma pack(push, 1)
typedef struct{
  uint32_t runTime;
  uint8_t posValid;
  uint8_t velValid;
  uint8_t system;
  uint8_t numSV;
  uint8_t numSVGPS;
  uint8_t numSVBDS;
  uint8_t numSVGLN;
  uint8_t res;
  float pDop;
  double lon;
  double lat;
  float height;
  float sepGeoid;
  float hAcc;
  float vAcc;
  float velN;
  float velE;
  float velU;
  float speed3D;
  float speed2D;
  float heading;
  float sAcc;
  float cAcc;
}NAV_PV;

#pragma pack(push, 1)
typedef struct{
  uint32_t runTime;
  float tAcc;
  float msErr;
  uint16_t ms;
  uint16_t year;
  uint8_t month;
  uint8_t day;
  uint8_t hour;
  uint8_t min;
  uint8_t sec;
  uint8_t valid;
  uint8_t timeSrc;
  uint8_t dateValid;
}NAV_TIMEUTC;

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

enum MES_STATE{
  HEADER,
  LENGTH,
  CLSID,
  SUBID,
  PAYLOAD,
  CHKSUM_CHK
};

void sendByteMes(const uint8_t* b_data, size_t mes_len, mwx::serial_jen* ser);
void sendStrMes(const char* c_data, size_t mes_len, mwx::serial_jen* ser);

bool casicParser(uint8_t b_data, uint8_t* clsid, uint8_t* sub_id, uint16_t* mes_len, uint32_t* chk_sum, uint8_t* payload);
uint32_t calcChksum(uint8_t clsid, uint8_t sub_id, uint16_t len, const uint8_t* payload);

void setNavPV(NAV_PV* dest, const uint8_t* src);
void setNavTimeUtc(NAV_TIMEUTC* dest, const uint8_t* src);

#endif