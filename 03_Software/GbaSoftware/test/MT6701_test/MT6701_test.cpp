#include <TWELITE>
#include "mt6701.h"

MT6701 angleSensor0;
MT6701 angleSensor1;

/*** the setup procedure (called on boot) */
void setup() {
    angleSensor0.init(0, &SPI);
    angleSensor1.init(1, &SPI);
}

/*** the loop procedure (called every event) */
void loop() {
    float angle0, angle1;
    MAG_STRENGTH mag_strength0, mag_strength1;
    PUSH_BOTTON push_botton0, push_botton1;
    TRACK track0, track1;

    angleSensor0.getAngle(&angle0);
    angleSensor0.getStatus(&mag_strength0, &push_botton0, &track0);

    angleSensor1.getAngle(&angle1);
    angleSensor1.getStatus(&mag_strength1, &push_botton1, &track1);

    Serial.print(millis());
    Serial.print("\t");
    
    Serial.print(angle0);
    Serial.print("\t");
    Serial.print(angle1);
    Serial.print("\t");

    Serial.print(mag_strength0);
    Serial.print("\t");
    Serial.print(mag_strength1);
    Serial.print("\t");

    Serial.print(push_botton0);
    Serial.print("\t");
    Serial.print(push_botton1);
    Serial.print("\t");

    Serial.print(track0);
    Serial.print("\t");
    Serial.println(track1);

    delay(10);

}