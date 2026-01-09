#include <TWELITE>
#include "BMP280.h"

BMP280 bmp280;

/*** the setup procedure (called on boot) */
void setup() {
    Serial.println("BMP280_initialize");
    Wire.begin(WIRE_CONF::WIRE_50KHZ, false);
    bmp280.init();
    Serial.println("BMP280_start");
}

/*** the loop procedure (called every event) */
void loop() {
    float temperature;
    uint32_t pressure;
    temperature = bmp280.getTemperature();
    pressure = bmp280.getPressure();

    Serial.print(temperature);
    Serial.print("\t");
    Serial.println(pressure);

    delay(100);

}