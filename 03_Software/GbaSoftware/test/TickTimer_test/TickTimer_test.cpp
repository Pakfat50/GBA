#include <TWELITE>
#include <math.h>
#include "utility.h"

#define TIME_INTERVAL 10000 // usec

uint32_t t_now, t_old, delta_t, delay_time;

/*** the setup procedure (called on boot) */
void setup() {
    uint32_t cyc = 1000 / sToCoNet_AppContext.u16TickHz;
    cyc *= 16000;
    Serial.println(cyc);
}

/*** the loop procedure (called every event) */
void loop() {
    t_old = micros();

    tan(0.1);
    Serial.println(t_old);

    delta_t = micros() - t_old;
    delay_time = TIME_INTERVAL-delta_t;

    if( (delay_time > 0) && (delay_time < TIME_INTERVAL)){
        delayMicroseconds(delay_time);
    }
}

