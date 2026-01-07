#include <TWELITE>
#include "gps_passing_mode_task.h"
#include "serial_parser.h"
#include "gba_mode.h"

void gpsPassingModeInit(mwx::serial_jen* gps_serial){
    
    gps_serial->setup(GPS_SERIAL_TX_BUF_SIZE, GPS_SERIAL_RX_BUF_SIZE);

#ifdef USE_ALT_PIN
    gps_serial->begin(GPS_SERIAL_BAUDRATE, uint8_t(serial_jen::E_CONF::PORT_ALT));
#else
    gps_serial->begin(GPS_SERIAL_BAUDRATE);
#endif

}

GBA_MODE gpsPassingModeTask(mwx::serial_jen* gps_serial, mwx::serial_jen* debug_serial){
    bool check_mode = false;
    uint8_t b_mode;
    GBA_MODE ret_mode;

    while(gps_serial->available()) {
        auto c = gps_serial->read();
        *debug_serial << char_t(c);
    }

    while(debug_serial->available()) {
        auto c = debug_serial->read();
        *gps_serial << char_t(c);

        if(parseMode(c, &b_mode) == GET_VALUE){
            check_mode = true;
        }
    }

    if(check_mode = true){
        ret_mode = mode_checker(b_mode, GPS_PASSING);
    }

    return ret_mode;
}
