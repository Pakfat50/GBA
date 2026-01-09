#include <TWELITE>
#include "casic_parser.h"

//#define DEBUG_

const uint32_t BAUD_UART1 = 9600;

uint32_t chk_sum = 0;
uint32_t mes_pos = 0;
uint16_t mes_len = 0;
uint8_t clsid = 0;
uint8_t sub_id = 0;
uint8_t payload[MAX_BUF_SIZE] = {0};
NAV_PV nav_sv;
NAV_TIMEUTC nav_timeutc;


/*** the setup procedure (called on boot) */
void setup() {
    // Serial.begin();

    // initialize the object. (allocate Tx/Rx buffer, and etc..)
    Serial1.setup(64, 192);

    // start the peripheral with 115200bps (DIO14,15)
    Serial1.begin(BAUD_UART1, uint8_t(serial_jen::E_CONF::PORT_ALT));

    Serial.println("GPS Initialize");
    
    sendStrMes(str_nema_pcsa04, sizeof(str_nema_pcsa04), &Serial1);
    delay(100);
    sendByteMes(mes_disable_nema, sizeof(mes_disable_nema), &Serial1);
    delay(100);
    sendByteMes(mes_enable_pv, sizeof(mes_enable_pv), &Serial1);
    delay(100);
    sendByteMes(mes_enable_timeutc, sizeof(mes_enable_timeutc), &Serial1);
    delay(100);

    Serial.println("GPS Start");

    // To assign alternative ports DIO11(TxD),09(RxD).
    //Serial1.begin(115200, uint8_t(serial_jen::E_CONF::PORT_ALT));
}

/*** the loop procedure (called every event) */
void loop() {
    while(Serial1.available()) {
        auto c_data = Serial1.read();
        uint8_t b_data = (uint8_t)c_data;

        if(casicParser(b_data, &clsid, &sub_id, &mes_len, &chk_sum, payload)==true){
            uint32_t calc_check_sum = calcChksum(clsid, sub_id, mes_len, payload);

            if(chk_sum == calc_check_sum){
                if((clsid==0x01)&&(sub_id==0x03)&&(mes_len==sizeof(nav_sv))){
                    setNavPV(&nav_sv, payload);
                }

                if((clsid==0x01)&&(sub_id==0x10)&&(mes_len==sizeof(nav_timeutc))){
                    setNavTimeUtc(&nav_timeutc, payload);
                }
            }

#ifdef DEBUG_
            Serial.println(mes_len);
            Serial.println(clsid, HEX);
            Serial.println(sub_id, HEX);
            for(int i=0; i<mes_len; i++){
                Serial.print(payload[i],HEX);
                Serial.print(" ");
            }
            Serial.println();
            Serial.print(chk_sum, HEX);
            Serial.println();
            Serial.print(calc_check_sum, HEX);
            Serial.println();
            Serial.println();
#endif 

            Serial.print(nav_timeutc.runTime);
            Serial.print(" ");
            Serial.print(nav_timeutc.year);
            Serial.print(" ");
            Serial.print(nav_timeutc.month);
            Serial.print(" ");
            Serial.print(nav_timeutc.day);
            Serial.print(" ");
            Serial.print(nav_timeutc.hour);
            Serial.print(" ");
            Serial.print(nav_timeutc.min);
            Serial.print(" ");
            Serial.println(nav_timeutc.sec);      
            
            Serial.print(nav_sv.runTime);
            Serial.print(" ");
            Serial.print(nav_sv.lon);
            Serial.print(" ");
            Serial.print(nav_sv.lat);
            Serial.print(" ");
            Serial.println(nav_sv.numSV);
        }
    }
}

