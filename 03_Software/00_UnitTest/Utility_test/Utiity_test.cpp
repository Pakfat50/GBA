#include <TWELITE>
#include "utility.h"

void setup() {

}

void begin(){

    Serial.println("Lowpass filter Test");
    LOWPASS lowpass;
    lowpass.init(0.02, 1);
    for(int i = 0; i<100; i++){
        float x = 0;
        Serial.print("x = ");
        Serial.print(x);
        Serial.print(" x_lowp = ");
        Serial.println(lowpass.get(x));
        delay(10);
    }

    for(int i = 0; i<100; i++){
        float x = 1;
        Serial.print("x = ");
        Serial.print(x);
        Serial.print(" x_lowp = ");
        Serial.println(lowpass.get(x));
        delay(10);
    }


    Serial.println("\n\nMoving Average Test");
    MOVING_AVERAGE moving_average;
    moving_average.init(10, 0.5);
    for(int i = 0; i<100; i++){
        float x = (float)i * 0.01;
        Serial.print("i = ");
        Serial.print(i);
        Serial.print(" x = ");
        Serial.print(x);
        Serial.print(" average = ");
        Serial.println(moving_average.get(x));
        delay(10);
    }

}

void loop() {

}

