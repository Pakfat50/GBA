#include <TWELITE>

const uint32_t BAUD_UART1 = 9600;

/*** the setup procedure (called on boot) */
void setup() {
    // Serial.begin();

    // initialize the object. (allocate Tx/Rx buffer, and etc..)
    Serial1.setup(64, 192);

    // start the peripheral with 115200bps (DIO14,15)
    Serial1.begin(BAUD_UART1, uint8_t(serial_jen::E_CONF::PORT_ALT));

    // To assign alternative ports DIO11(TxD),09(RxD).
    //Serial1.begin(115200, uint8_t(serial_jen::E_CONF::PORT_ALT));
}

/*** the loop procedure (called every event) */
void loop() {
    while(Serial1.available()) {
        auto c = Serial1.read();
        Serial << char_t(c);
    }

    while(Serial.available()) {
        auto c = Serial.read();
        Serial1 << char_t(c);
    }
}

