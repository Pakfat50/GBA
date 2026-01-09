#include <TWELITE>
#include "serial_parser.h"
#include "gba_common.h"

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
        
        switch (parseMode(b_data, &mode)){
            case GET_VALUE:
                Serial.print("MODE:");
                Serial.write(mode);
                Serial.println();
                Serial.println();
                break;
            
            case ERR_CRLF:
                Serial.println("CRLF is not matched");
                break;
            
            default:
                break;
        }


        switch (parseVal(b_data, &param_num, &val)){
            case GET_VALUE:
                Serial.print("PARAM NUM:");
                Serial.print(param_num);
                Serial.println();
                Serial.print("VALUE:");
                Serial.println(val);
                Serial.println();
                break;
            
            case ERR_INVALID_ID:
                Serial.println("PARAM_NUM inputs is invalid");
                break;

            case ERR_INVALID_VAL:
                Serial.println("Val inputs is invalid");
                break;    

            case ERR_OVER_RANGE:
                Serial.print("PARAM_NUM inputs overrange (Max size is ");
                Serial.print(PARAM_NUM_MAX);
                Serial.println(")");            
                break;

            default:
                break;
        }
    }
}
