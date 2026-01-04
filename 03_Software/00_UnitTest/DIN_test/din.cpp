#include <TWELITE>
#include "din.h"

void dinInit(void){
    pinMode(DIN_PIN0, PIN_MODE::INPUT); //DIN0
    pinMode(DIN_PIN1, PIN_MODE::INPUT); //DIN1
    pinMode(DIN_PIN2, PIN_MODE::INPUT); //DIN2
    pinMode(DIN_PIN3, PIN_MODE::INPUT); //DIN3
    pinMode(DIN_PIN4, PIN_MODE::INPUT); //DIN4
    pinMode(DIN_PIN5, PIN_MODE::INPUT); //DIN5
    pinMode(DIN_PIN6, PIN_MODE::INPUT); //DIN6
    pinMode(DIN_PIN7, PIN_MODE::INPUT); //DIN7
}

uint8_t getChannel(void){
    uint8_t din0, din1, din2, din3;
    uint8_t channel = 0;

    din0 = (digitalRead(DIN_PIN0) == HIGH) ? 1: 0;
    din1 = (digitalRead(DIN_PIN1) == HIGH) ? 1: 0;
    din2 = (digitalRead(DIN_PIN2) == HIGH) ? 1: 0;
    din3 = (digitalRead(DIN_PIN3) == HIGH) ? 1: 0;

    channel  = din3;
    channel |= din2<<1;
    channel |= din1<<2;
    channel |= din0<<3;

    return channel;

}

uint8_t getSensorId(void){
    uint8_t din4, din5, din6, din7;
    uint8_t sensor_id = 0;

    din4 = (digitalRead(DIN_PIN4) == HIGH) ? 1: 0;
    din5 = (digitalRead(DIN_PIN5) == HIGH) ? 1: 0;
    din6 = (digitalRead(DIN_PIN6) == HIGH) ? 1: 0;
    din7 = (digitalRead(DIN_PIN7) == HIGH) ? 1: 0;

    sensor_id  = din7;
    sensor_id |= din6<<1;
    sensor_id |= din5<<2;
    sensor_id |= din4<<3;

    return sensor_id;
}