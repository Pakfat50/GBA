#include <TWELITE>
#include "logger.h"
#include "gba_common.h"

static uint32_t calc_checksum(uint8_t clsid, uint8_t sub_id, uint16_t len, uint8_t payload[]);

void send_data(uint8_t clsid, uint8_t sub_id, uint16_t len, uint8_t payload[], mwx::serial_jen* ser){
    U1_U2 l_len;
    U1_U4 l_checksum;
    
    l_len.U2 = len;
    l_checksum.U4 = calc_checksum(clsid, sub_id, len, payload);

    ser->write(header[0]);
    ser->write(header[1]);
    ser->write(header[2]);
    ser->write(header[3]);
    ser->write(clsid);
    ser->write(sub_id);
    ser->write(l_len.U1[0]);
    ser->write(l_len.U1[1]);

    for(size_t i=0; i<len; i++){
        ser->write(payload[i]);
    }

    ser->write(l_checksum.U1[0]);
    ser->write(l_checksum.U1[1]);    
    ser->write(l_checksum.U1[2]);
    ser->write(l_checksum.U1[3]);
    
}

static uint32_t calc_checksum(uint8_t clsid, uint8_t sub_id, uint16_t len, uint8_t payload[]){
    U1_U2 l_len;
    uint32_t checksum = 0;

    l_len.U2 = len;

    checksum += (uint32_t)clsid;
    checksum += (uint32_t)sub_id;
    checksum += (uint32_t)l_len.U1[0];
    checksum += (uint32_t)l_len.U1[1];

    for(size_t i=0; i<len; i++){
        checksum += (uint32_t)payload[i];
    }

    return checksum;
}