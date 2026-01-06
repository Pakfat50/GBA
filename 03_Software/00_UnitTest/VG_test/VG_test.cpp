#include <TWELITE>
#include <stdio.h>
#include "madgwickFilter.h"

void setup() {

}

void begin() {
    
    char buf[100];

    // flat upright position to resolve the graident descent problem
    Serial.print("\n\n Upright to solve gradient descent problem\n");
    for(float i = 0; i<1000; i++){
        imu_filter(0.05, 0.05, 0.9, 0, 0, 0);
        eulerAngles(&roll, &pitch, &yaw);
        sprintf(buf, "Time (s): %.3f, roll: %f, pitch: %f, yaw: %f\n", i * DELTA_T, roll, pitch, yaw);
        Serial.print(buf);
    }

    // Angular Rotation around Z axis (yaw) at 100 degrees per second
    Serial.print("\n\nYAW 100 Degrees per Second\n");
    for(float i = 0; i<1000; i++){
        imu_filter(0, 0, 1, 0, 0, -100 * PI / 180);
        eulerAngles(&roll, &pitch, &yaw);
        sprintf(buf, "Time (s): %.3f, roll: %f, pitch: %f, yaw: %f\n", i * DELTA_T, roll, pitch, yaw);
        Serial.print(buf);
    }

    // Angular Rotation around Z axis (yaw) at -20 degrees per second
    Serial.print("\n\nYAW -20 Degrees per Second\n");
    for(float i = 0; i<1000; i++){
        imu_filter(0, 0, 1, 0, 0, 20 * PI / 180);
        eulerAngles(&roll, &pitch, &yaw);
        sprintf(buf, "Time (s): %.3f, roll: %f, pitch: %f, yaw: %f\n", i * DELTA_T, roll, pitch, yaw);
        Serial.print(buf);
    }
    
}

void loop() {

}

