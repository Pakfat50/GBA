#include <TWELITE>
#include "casic_parser.h"
#include "gba_common.h"

void sendByteMes(const uint8_t* b_data, size_t mes_len, mwx::serial_jen* ser){
  for (size_t i = 0; i < mes_len; i++)
  {
    ser->write(b_data[i]);
  }
}

void sendStrMes(const char* c_data, size_t mes_len, mwx::serial_jen* ser){
  for (size_t i = 0; i < mes_len; i++)
  {
    ser->putchar(c_data[i]);
  }
}

bool casicParser(uint8_t b_data, uint8_t* clsid, uint8_t* sub_id, uint16_t* mes_len, uint32_t* chk_sum, uint8_t* payload){
  bool l_res = false;

  static MES_STATE s_mes_state =  HEADER;
  static uint32_t s_chk_sum = 0;
  static uint16_t s_mes_pos = 0;
  static uint16_t s_mes_len = 0;
  static uint8_t s_clsid = 0;
  static uint8_t s_sub_id = 0;
  static uint8_t s_payload[MAX_BUF_SIZE] = {0};
  static uint8_t s_chksum_byte[4] = {0};


  switch (s_mes_state){
    case HEADER:
      if ((s_mes_pos==0)&&(b_data == (uint8_t)0xBA)){
        s_mes_pos +=1;
      }
      else if((s_mes_pos==1)&&(b_data == (uint8_t)0xCE)){
        s_mes_pos = 0;
        s_mes_state = LENGTH;
      }
      else{
        s_chk_sum = 0;
        s_mes_pos = 0;
        s_mes_len = 0;
        s_clsid = 0;
        s_sub_id = 0;
      }
      break;
    case LENGTH:
      if(s_mes_pos == 0){
        s_mes_pos += 1;
        s_mes_len = (uint16_t)b_data;
      }
      else{
        s_mes_pos = 0;
        s_mes_len += (uint16_t)b_data<<8;
        s_mes_state = CLSID;
      }
      break;
    case CLSID:
        s_clsid = b_data;
        s_mes_state = SUBID;
      break;
    case SUBID:
        s_sub_id = b_data;
        s_mes_state = PAYLOAD;
      break;
    case PAYLOAD:
      if((s_mes_pos < s_mes_len-1)&&(s_mes_pos < MAX_BUF_SIZE-1)){
        s_payload[s_mes_pos] = b_data;
        s_mes_pos += 1;
      }
      else{
        s_payload[s_mes_pos] = b_data;
        s_mes_pos = 0;
        s_mes_state = CHKSUM_CHK;
      }
      break;
    case CHKSUM_CHK:
      if((s_mes_pos < 3)){
        s_chksum_byte[s_mes_pos] = b_data;
        s_mes_pos += 1;
      }
      else{
        s_mes_pos = 0;
        s_chksum_byte[3] = b_data;
        s_chk_sum  = ((uint32_t)s_chksum_byte[3])<<24;
        s_chk_sum += ((uint32_t)s_chksum_byte[2])<<16;
        s_chk_sum += ((uint32_t)s_chksum_byte[1])<<8;
        s_chk_sum += ((uint32_t)s_chksum_byte[0]);
        s_mes_state = HEADER;

        *clsid = s_clsid;
        *sub_id = s_sub_id;
        *mes_len = s_mes_len;
        *chk_sum = s_chk_sum;
        for(int i=0; i<s_mes_len; i++){
            payload[i] = s_payload[i];
        }
        l_res = true;
      }
      break;
  }
  
  return l_res;
}

uint32_t calcChksum(uint8_t clsid, uint8_t sub_id, uint16_t len, const uint8_t* payload){
    uint32_t ckSum = 0;
    ckSum  = ((uint32_t)sub_id)<<24;
    ckSum += ((uint32_t)clsid)<<16;
    ckSum += (uint32_t)len;
    
    for(int i=0; i<(len/4); i++){
        uint32_t l_payload = 0;
        
        l_payload  = ((uint32_t)payload[i*4+3])<<24;
        l_payload += ((uint32_t)payload[i*4+2])<<16;
        l_payload += ((uint32_t)payload[i*4+1])<<8;
        l_payload += ((uint32_t)payload[i*4]);

        ckSum += l_payload;
    }
    return ckSum;
}

void setNavPV(NAV_PV* dest, const uint8_t* src){
  U1_U4 u1_u4;
  U1_R4 u1_r4;
  U1_R8 u1_r8;

  u1_u4.U1[3] = src[0];
  u1_u4.U1[2] = src[1];
  u1_u4.U1[1] = src[2];
  u1_u4.U1[0] = src[3];
  dest->runTime = u1_u4.U4;

  dest->posValid = src[4];
  dest->velValid = src[5];
  dest->system = src[6];
  dest->numSV = src[7];
  dest->numSVGPS = src[8];
  dest->numSVBDS = src[9];
  dest->numSVGLN = src[10];
  dest->res = src[11];

  u1_r4.U1[3] = src[12];
  u1_r4.U1[2] = src[13];
  u1_r4.U1[1] = src[14];
  u1_r4.U1[0] = src[15];
  dest->pDop = u1_r4.R4;

  u1_r8.U1[7] = src[16];
  u1_r8.U1[6] = src[17];
  u1_r8.U1[5] = src[18];
  u1_r8.U1[4] = src[19];
  u1_r8.U1[3] = src[20];
  u1_r8.U1[2] = src[21];
  u1_r8.U1[1] = src[22];
  u1_r8.U1[0] = src[23];
  dest->lon = u1_r8.R8;

  u1_r8.U1[7] = src[24];
  u1_r8.U1[6] = src[25];
  u1_r8.U1[5] = src[26];
  u1_r8.U1[4] = src[27];
  u1_r8.U1[3] = src[28];
  u1_r8.U1[2] = src[29];
  u1_r8.U1[1] = src[30];
  u1_r8.U1[0] = src[31];
  dest->lat = u1_r8.R8;

  u1_r4.U1[3] = src[32];
  u1_r4.U1[2] = src[33];
  u1_r4.U1[1] = src[34];
  u1_r4.U1[0] = src[35];  
  dest->height = u1_r4.R4;

  u1_r4.U1[3] = src[36];
  u1_r4.U1[2] = src[37];
  u1_r4.U1[1] = src[38];
  u1_r4.U1[0] = src[39];  
  dest->sepGeoid = u1_r4.R4;

  u1_r4.U1[3] = src[40];
  u1_r4.U1[2] = src[41];
  u1_r4.U1[1] = src[42];
  u1_r4.U1[0] = src[43]; 
  dest->hAcc = u1_r4.R4;

  u1_r4.U1[3] = src[44];
  u1_r4.U1[2] = src[45];
  u1_r4.U1[1] = src[46];
  u1_r4.U1[0] = src[47]; 
  dest->vAcc = u1_r4.R4;

  u1_r4.U1[3] = src[48];
  u1_r4.U1[2] = src[49];
  u1_r4.U1[1] = src[50];
  u1_r4.U1[0] = src[51]; 
  dest->velN = u1_r4.R4;

  u1_r4.U1[3] = src[52];
  u1_r4.U1[2] = src[53];
  u1_r4.U1[1] = src[54];
  u1_r4.U1[0] = src[55]; 
  dest->velE = u1_r4.R4;

  u1_r4.U1[3] = src[56];
  u1_r4.U1[2] = src[57];
  u1_r4.U1[1] = src[58];
  u1_r4.U1[0] = src[59]; 
  dest->velU = u1_r4.R4;

  u1_r4.U1[3] = src[60];
  u1_r4.U1[2] = src[61];
  u1_r4.U1[1] = src[62];
  u1_r4.U1[0] = src[63]; 
  dest->speed3D = u1_r4.R4;

  u1_r4.U1[3] = src[64];
  u1_r4.U1[2] = src[65];
  u1_r4.U1[1] = src[66];
  u1_r4.U1[0] = src[67]; 
  dest->speed2D = u1_r4.R4;

  u1_r4.U1[3] = src[68];
  u1_r4.U1[2] = src[69];
  u1_r4.U1[1] = src[70];
  u1_r4.U1[0] = src[71]; 
  dest->heading = u1_r4.R4;

  u1_r4.U1[3] = src[72];
  u1_r4.U1[2] = src[73];
  u1_r4.U1[1] = src[74];
  u1_r4.U1[0] = src[75]; 
  dest->sAcc = u1_r4.R4;

  u1_r4.U1[3] = src[76];
  u1_r4.U1[2] = src[77];
  u1_r4.U1[1] = src[78];
  u1_r4.U1[0] = src[79]; 
  dest->cAcc = u1_r4.R4;
}

void setNavTimeUtc(NAV_TIMEUTC* dest, const uint8_t* src){
  U1_U4 u1_u4;
  U1_U2 u1_u2;
  U1_R4 u1_r4;

  u1_u4.U1[3] = src[0];
  u1_u4.U1[2] = src[1];
  u1_u4.U1[1] = src[2];
  u1_u4.U1[0] = src[3];
  dest->runTime = u1_u4.U4;

  u1_r4.U1[3] = src[4];
  u1_r4.U1[2] = src[5];
  u1_r4.U1[1] = src[6];
  u1_r4.U1[0] = src[7]; 
  dest->tAcc = u1_r4.R4;

  u1_r4.U1[3] = src[8];
  u1_r4.U1[2] = src[9];
  u1_r4.U1[1] = src[10];
  u1_r4.U1[0] = src[11]; 
  dest->msErr = u1_r4.R4;

  u1_u2.U1[1] = src[12];
  u1_u2.U1[0] = src[13];
  dest->ms = u1_u2.U2;

  u1_u2.U1[1] = src[14];
  u1_u2.U1[0] = src[15];
  dest->year = u1_u2.U2;

  dest->month = src[16];
  dest->day = src[17];
  dest->hour = src[18];
  dest->min = src[19];
  dest->sec = src[20];
  dest->valid = src[21];
  dest->timeSrc = src[22];
  dest->dateValid = src[23];
}