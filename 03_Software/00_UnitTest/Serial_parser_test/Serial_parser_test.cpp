#include <TWELITE>
#include "serial_parser.h"

uint8_t mode = 0;
uint16_t param_num = 0;
float val = 0;

/*** the setup procedure (called on boot) */
void setup() {

}

/*** the loop procedure (called every event) */
void loop() {
    while(Serial.available()) {
        auto c_data = Serial.read();
        uint8_t b_data = (uint8_t)c_data;

        Serial.write(c_data);
        
        if(parseMode(b_data, &mode) == true){
            Serial.print("MODE:");
            Serial.write(mode);
            Serial.println();
            Serial.println();
        }

        if(parseVal(b_data, &param_num, &val) == true){
            Serial.print("PARAM NUM:");
            Serial.print(param_num);
            Serial.println();
            Serial.print("VALUE:");
            Serial.println(val);
            Serial.println();
        }
    }
}
