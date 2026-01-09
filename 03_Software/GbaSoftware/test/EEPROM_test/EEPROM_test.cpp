#include <TWELITE>
#include "eeprom.h"

#define DIN_PIN0 2

#define VAL1_NUM 0
#define VAL2_NUM 1

void setup() {
    float val1 = (float)random(0, 1000)*0.001 * 10;
    float val2 = (float)random(0, 900)*0.001 * 10;

    pinMode(DIN_PIN0, PIN_MODE::INPUT);

    if(digitalRead(DIN_PIN0) == HIGH){
        eeprom_float_wite(VAL1_NUM, val1);
        eeprom_float_wite(VAL2_NUM, val2);          
        Serial.println("EEPROM UPDATED");
        Serial.print("Written value: val1 = ");
        Serial.print(val1);
        Serial.print(" val2 = ");
        Serial.println(val2);
    }
    else{
        Serial.println("EEPROM DIDNOT UPDATE. READ ONLY");
    }
}

void loop() {
    float val1_read;
    float val2_read;

    eeprom_float_read(VAL1_NUM, &val1_read);
    eeprom_float_read(VAL2_NUM, &val2_read);

    Serial.print(val1_read);
    Serial.print("\t");
    Serial.println(val2_read);

    delay(1000);
}
