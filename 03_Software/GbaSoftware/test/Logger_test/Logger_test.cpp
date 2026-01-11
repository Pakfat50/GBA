#include <TWELITE>
#include <stdio.h>
#include <string.h>
#include "logger.h"



#define CLASS 0x01
#define ANGLE_ID 0x02

#pragma pack(push, 1)
typedef struct{
    uint32_t time;
    float angle1;
    float angle2;
    uint8_t id;
}ANGLE;
#pragma pack(pop)

ANGLE angle;
uint8_t payload[sizeof(ANGLE)] = {0};

/*** the setup procedure (called on boot) */
void setup() {
    angle.time = 0;
    angle.angle1 = 12.4;
    angle.angle2 = 23.5;
    angle.id = 0x20;
}

/*** the loop procedure (called every event) */
void loop() {
    angle.time = (uint32_t)millis();
    memcpy(payload, &angle, sizeof(ANGLE));
    
    send_data(CLASS, ANGLE_ID, (uint16_t)sizeof(ANGLE), payload, &Serial);

    delay(1000);

}

