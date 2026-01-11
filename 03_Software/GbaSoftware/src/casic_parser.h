#ifndef CASIC_PARSER_H_
#define CASIC_PARSER_H_

#include <TWELITE>
#include "gba_common.h"

#define GPS_MAX_BUF_SIZE 120
#define USE_ALT_PIN 
#define GPS_SERIAL_BAUDRATE 9600
#define GPS_SERIAL_TX_BUF_SIZE 64
#define GPS_SERIAL_RX_BUF_SIZE 192

#define GPS_NAV_CLSID 0x01
#define GPS_NAVSV_ID 0x03
#define GPS_NAVTIMEUTC_ID 0x10

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
#pragma pack(pop)

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
#pragma pack(pop)

enum MES_STATE{
  HEADER,
  LENGTH,
  CLSID,
  SUBID,
  PAYLOAD,
  CHKSUM_CHK
};

void gpsSerialInit(mwx::serial_jen* ser);
void sendByteMes(const uint8_t* b_data, size_t mes_len, mwx::serial_jen* ser);
void sendStrMes(const char* c_data, size_t mes_len, mwx::serial_jen* ser);

bool casicParser(uint8_t b_data, uint8_t* clsid, uint8_t* sub_id, uint16_t* mes_len, uint32_t* chk_sum, uint8_t* payload);
uint32_t calcChksum(uint8_t clsid, uint8_t sub_id, uint16_t len, const uint8_t* payload);

void setNavPV(NAV_PV* dest, const uint8_t* src);
void setNavTimeUtc(NAV_TIMEUTC* dest, const uint8_t* src);

#endif