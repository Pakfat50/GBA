#include <TWELITE>
#include <math.h>
#include "utility.h"

#define PI 3.14159265

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
    Serial.print("x = ");
    Serial.print(x);
    Serial.print("\tmax = ");
    Serial.print(min_max.getMax());
    Serial.print("\tmin = ");
    Serial.print(min_max.getMin());
    Serial.print("\tMaxNorm = ");
    Serial.println(min_max.getMaxNorm());

    x = 2;
    min_max.set(x);
    Serial.print("x = ");
    Serial.print(x);
    Serial.print("\tmax = ");
    Serial.print(min_max.getMax());
    Serial.print("\tmin = ");
    Serial.print(min_max.getMin());
    Serial.print("\tMaxNorm = ");
    Serial.println(min_max.getMaxNorm());

    x = -3;
    min_max.set(x);
    Serial.print("x = ");
    Serial.print(x);
    Serial.print("\tmax = ");
    Serial.print(min_max.getMax());
    Serial.print("\tmin = ");
    Serial.print(min_max.getMin());
    Serial.print("\tMaxNorm = ");
    Serial.println(min_max.getMaxNorm());

    Serial.println("reset");
    min_max.reset(x);
    Serial.print("x = ");
    Serial.print(x);
    Serial.print("\tmax = ");
    Serial.print(min_max.getMax());
    Serial.print("\tmin = ");
    Serial.print(min_max.getMin());   
    Serial.print("\tMaxNorm = ");
    Serial.println(min_max.getMaxNorm());


    Serial.println("\n\nRange Check Test");
    x = 1;
    bool err = false;
    Serial.print("max = 10, min = 0, default = 5, x = ");
    Serial.print(x);
    err = range_check(&x, 10, 0, 5);
    Serial.print(" err = ");
    Serial.print(err);
    Serial.print(" ret_x = ");
    Serial.println(x);

    x = 100;
    Serial.print("max = 10, min = 0, default = 5, x = ");
    Serial.print(x);
    err = range_check(&x, 10, 0, 5);
    Serial.print(" err = ");
    Serial.print(err);
    Serial.print(" ret_x = ");
    Serial.println(x);

    x = -100;
    Serial.print("max = 10, min = 0, default = 5, x = ");
    Serial.print(x);
    err = range_check(&x, 10, 0, 5);
    Serial.print(" err = ");
    Serial.print(err);
    Serial.print(" ret_x = ");
    Serial.println(x);

    x = sqrt(-1); //NaN
    Serial.print("max = 10, min = 0, default = 5, x = ");
    Serial.print(x);
    err = range_check(&x, 10, 0, 5);
    Serial.print(" err = ");
    Serial.print(err);
    Serial.print(" ret_x = ");
    Serial.println(x);

    x = 1.0/0.0; //Inf
    Serial.print("max = 10, min = 0, default = 5, x = ");
    Serial.print(x);
    err = range_check(&x, 10, 0, 5);
    Serial.print(" err = ");
    Serial.print(err);
    Serial.print(" ret_x = ");
    Serial.println(x);

    Serial.println("\n\ntan Test");
    Serial.print("tan(pi/4) = ");
    Serial.println(tan(PI/4));
    Serial.print("tan(pi/2) = ");
    Serial.println(tan(PI/2));
    Serial.print("tan(pi/4 + 2pi) = ");
    Serial.println(tan(PI/4 + 2*PI));
    Serial.print("tan(pi/4 - 3pi) = ");
    Serial.println(tan(PI/4 - 3*PI));


}

void loop() {

}

