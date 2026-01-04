#include <TWELITE>
#include <NWK_SIMPLE>
#include "gba_twenet.h"

#define SENSOR_ID 0x00
#define CHANNEL 13

GBA_DATA gba_data;
RX_INFO rx_info;

void setup() {
	gbaTwenetInit(SENSOR_ID, CHANNEL);
}

void loop() {

}

void on_rx_packet(packet_rx& rx, bool_t &handled) {
	gbaTwenetReceive(&gba_data, &rx_info, &rx);
	handled = true;

    Serial.print(gba_data.year);
	Serial.print("\t");
    Serial.print(gba_data.month);
	Serial.print("\t");
    Serial.print(gba_data.day);
	Serial.print("\t");
    Serial.print(gba_data.hour);
	Serial.print("\t");
    Serial.print(gba_data.min);
	Serial.print("\t");
    Serial.print(gba_data.sec);
	Serial.print("\t");
    Serial.print(gba_data.ms);
	Serial.print("\t");
    Serial.print(gba_data.averageWindSpeedE);
	Serial.print("\t");
    Serial.print(gba_data.averagewindSpeedN);
	Serial.print("\t");
    Serial.print(gba_data.gustWindSpeedE);
	Serial.print("\t");
    Serial.print(gba_data.gustWindSpeedN);
	Serial.print("\t");
    Serial.print(gba_data.lon);
	Serial.print("\t");
    Serial.print(gba_data.lat);
	Serial.print("\t");
    Serial.print(gba_data.numSV);
	Serial.print("\t");
    Serial.print(gba_data.temperature);
	Serial.print("\t");
    Serial.print(gba_data.humidity);
	Serial.print("\t");
    Serial.print(gba_data.pressure);
	Serial.print("\t");
	Serial.print(rx_info.srcAddr, HEX);
	Serial.print("\t");
	Serial.print(rx_info.seq);
	Serial.print("\t");
	Serial.println(rx_info.lqi);	
}
