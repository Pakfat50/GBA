#include <TWELITE>
#include "calc_wind_speed.h"


void setup() {

}

void begin(){
    float coff_angle = 0.04174;


    Serial.println("Calcuration Test");
    Serial.print("Coff angle = ");
    Serial.println(coff_angle);

    for(int i=-70; i<=70; i++){
        Serial.print("angle[deg] = ");
        Serial.print(i);
        Serial.print("\tWind speed[m/s] = ");
        Serial.println(calcWindSpeed( (float)i, coff_angle));
        delay(10);
    }

    uint32_t calc_num = 1000;
    Serial.println("\n\nCalcuration speed test");
    Serial.print("Caluration num = ");
    Serial.println(calc_num);

    uint32_t t_old = millis();
    for(uint32_t i = 0; i<calc_num; i++){
        calcWindSpeed( 10 , coff_angle);
    }
    uint32_t t_delta = millis() - t_old;
    float t_average = (float)t_delta /(float)calc_num;
    Serial.print("Average time[ms] = ");
    Serial.println(t_average);

}

void loop() {

}

