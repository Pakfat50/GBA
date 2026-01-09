#ifndef GBA_TWENET_H_
#define GBA_TWENET_H_

#include <TWELITE>
#include <stdio.h>

#define APP_ID 0x23474241 // #GBA
#define MES_LEN sizeof(GBA_DATA)
#define REPEAT_MAX 0
#define RETRY_NUM 0
#define TX_DELAY_ST 0
#define TX_DELAY_ED 0
#define TX_DELAY_INT 0

#pragma pack(push,1)
typedef struct{
    uint16_t year;
    uint8_t month;
    uint8_t day;
    uint8_t hour;
    uint8_t min;
    uint8_t sec;
    uint16_t ms;
    float averageWindSpeedE;
    float averagewindSpeedN;
    float gustWindSpeedE;
    float gustWindSpeedN;
    double lon;
    double lat;
    uint8_t numSV;
    float temperature;
    float humidity;
    float pressure;
}GBA_DATA;
#pragma pack(pop)

typedef struct{
    uint32_t receiveTime;
    uint8_t cmd;
    uint8_t srcAddr;
    uint8_t seq;
    uint8_t lqi;
}RX_INFO;

void gbaTwenetInit(uint8_t id, uint8_t channel, mwx::twenet* twelite);
bool gbaTwenetTransmit(GBA_DATA* gba_data, uint32_t addr, mwx::twenet* twelite);
void gbaTwenetReceive(GBA_DATA* gba_data, RX_INFO* rx_info, mwx::packet_rx* pkt);

#endif