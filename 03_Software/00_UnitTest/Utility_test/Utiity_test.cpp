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

    Serial.println("\n\nMin Max Test");
    MIN_MAX min_max;
    float x = 1;
    min_max.init(x);
    Serial.print("\tx = ");
    Serial.print(x);
    Serial.print("\tmax = ");
    Serial.print(min_max.getMax());
    Serial.print("\tmin = ");
    Serial.print(min_max.getMin());
    Serial.print("\tMaxNorm = ");
    Serial.print(min_max.getMaxNorm());

    x = 2;
    min_max.set(x);
    Serial.print("\tx = ");
    Serial.print(x);
    Serial.print("\tmax = ");
    Serial.print(min_max.getMax());
    Serial.print("\tmin = ");
    Serial.print(min_max.getMin());
    Serial.print("\tMaxNorm = ");
    Serial.print(min_max.getMaxNorm());

    x = -3;
    min_max.set(x);
    Serial.print("\tx = ");
    Serial.print(x);
    Serial.print("\tmax = ");
    Serial.print(min_max.getMax());
    Serial.print("\tmin = ");
    Serial.print(min_max.getMin());
    Serial.print("\tMaxNorm = ");
    Serial.print(min_max.getMaxNorm());

    Serial.println("reset");
    min_max.reset(x);
    Serial.print("\tx = ");
    Serial.print(x);
    Serial.print("\tmax = ");
    Serial.print(min_max.getMax());
    Serial.print("\tmin = ");
    Serial.print(min_max.getMin());   
    Serial.print("\tMaxNorm = ");
    Serial.print(min_max.getMaxNorm());

}

void loop() {

}

