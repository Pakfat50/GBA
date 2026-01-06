#include <TWELITE>
#include "madgwickFilter.h"

void setup() {

}

void begin() {
    float roll = 0.0, pitch = 0.0, yaw = 0.0;

    // flat upright position to resolve the graident descent problem
    Serial.print("\n\n Upright to solve gradient descent problem\n");
    for(float i = 0; i<300; i++){
        imu_filter(0.05, 0.05, 0.9, 0, 0, 0);
        eulerAngles(&roll, &pitch, &yaw);
        Serial.print("Time (s): ");
        Serial.print(i * DELTA_T);
        Serial.print("\troll:");
        Serial.print(roll);
        Serial.print("\tpitch:");
        Serial.print(pitch);       
        Serial.print("\tyaw:");
        Serial.println(yaw);
        delay(10);
    }

    // Angular Rotation around Z axis (yaw) at 100 degrees per second
    Serial.print("\n\nYAW 100 Degrees per Second\n");
    for(float i = 0; i<300; i++){
        imu_filter(0, 0, 1, 0, 0, -100 * PI / 180);
        eulerAngles(&roll, &pitch, &yaw);
        Serial.print("Time (s): ");
        Serial.print(i * DELTA_T);
        Serial.print("\troll:");
        Serial.print(roll);
        Serial.print("\tpitch:");
        Serial.print(pitch);       
        Serial.print("\tyaw:");
        Serial.println(yaw);
        delay(10);
    }

    // Angular Rotation around Z axis (yaw) at -20 degrees per second
    Serial.print("\n\nYAW -20 Degrees per Second\n");
    for(float i = 0; i<300; i++){
        imu_filter(0, 0, 1, 0, 0, 20 * PI / 180);
        eulerAngles(&roll, &pitch, &yaw);
        Serial.print("Time (s): ");
        Serial.print(i * DELTA_T);
        Serial.print("\troll:");
        Serial.print(roll);
        Serial.print("\tpitch:");
        Serial.print(pitch);       
        Serial.print("\tyaw:");
        Serial.println(yaw);
        delay(10);
    }
    
}

void loop() {

}

