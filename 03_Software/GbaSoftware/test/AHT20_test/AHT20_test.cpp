#include <TWELITE>
#include "AHT20.h"


/*** the setup procedure (called on boot) */
void setup() {
    Serial.println("AHT20_initialize");
    Wire.begin(WIRE_CONF::WIRE_50KHZ, false);
    aht20_begin();
    Serial.println("AHT20_start");
}

/*** the loop procedure (called every event) */
void loop() {
    float humidity, temperature;

    aht20_startSensor();

    if(aht20_getSensor(&humidity, &temperature)){

        Serial.print(humidity);
        Serial.print("\t");
        Serial.println(temperature);
    }
    delay(200);

}