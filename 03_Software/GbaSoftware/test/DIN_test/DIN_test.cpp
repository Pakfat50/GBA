#include <TWELITE>
#include "din.h"

/*** the setup procedure (called on boot) */
void setup() {
    dinInit();
}

/*** the loop procedure (called every event) */
void loop() {
    uint8_t channel = getChannel();
    uint8_t sensor_id = getSensorId();

    Serial.print(channel);
    Serial.print("\t");
    Serial.println(sensor_id);

    delay(100);
}

