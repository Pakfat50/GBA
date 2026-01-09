#include <TWELITE>
#include "gps_passing_mode_task.h"
#include "serial_parser.h"
#include "gba_mode.h"

void gpsPassingModeInit(void){
    
    GPS_SERIAL.setup(GPS_SERIAL_TX_BUF_SIZE, GPS_SERIAL_RX_BUF_SIZE);

#ifdef USE_ALT_PIN
    GPS_SERIAL.begin(GPS_SERIAL_BAUDRATE, uint8_t(serial_jen::E_CONF::PORT_ALT));
#else
    GPS_SERIAL.begin(GPS_SERIAL_BAUDRATE);
#endif

}

GBA_MODE gpsPassingModeTask(void){
    bool check_mode = false;
    uint8_t b_mode;
    GBA_MODE ret_mode;

    while(GPS_SERIAL.available()) {
        auto c = GPS_SERIAL.read();
        DEBUG_SERIAL << char_t(c);
    }

    while(DEBUG_SERIAL.available()) {
        auto c = DEBUG_SERIAL.read();
        GPS_SERIAL << char_t(c);

        if(parseMode(c, &b_mode) == GET_VALUE){
            check_mode = true;
        }
    }

    if(check_mode = true){
        ret_mode = mode_checker(b_mode, GPS_PASSING);
    }

    return ret_mode;
}
