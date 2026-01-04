#include <TWELITE>
#include <NWK_SIMPLE>
#include "gba_twenet.h"

#define SENSOR_ID 0x02
#define CHANNEL 13

GBA_DATA gba_data;

void setup() {
	gbaTwenetInit(SENSOR_ID, CHANNEL);
}

void loop() {
    uint32_t time = millis();

    Serial.print("Transmit! at ");
    Serial.println(time);
    
    gba_data.year = 2025;
    gba_data.month = 1;
    gba_data.day = 2;
    gba_data.hour = 15;
    gba_data.min = 16;
    gba_data.sec = (int)(time/1000);
    gba_data.ms = time - gba_data.sec*1000;
    gba_data.averageWindSpeedE = 5.2;
    gba_data.averagewindSpeedN = 2.4;
    gba_data.gustWindSpeedE = 6.3;
    gba_data.gustWindSpeedN = 4.5;
    gba_data.lon = 135.5;
    gba_data.lat = 43.5;
    gba_data.numSV = 10;
    gba_data.temperature = 20.2;
    gba_data.humidity = 0.58;
    gba_data.pressure = 1013.5;

	gbaTwenetTransmit(&gba_data, 0xFF);
	delay(200);
}

