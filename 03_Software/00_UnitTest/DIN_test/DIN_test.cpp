#include <TWELITE>

#define DIN_PIN0 2
#define DIN_PIN1 3
#define DIN_PIN2 12
#define DIN_PIN3 10
#define DIN_PIN4 16
#define DIN_PIN5 17
#define DIN_PIN6 5
#define DIN_PIN7 4

/*** the setup procedure (called on boot) */
void setup() {
    pinMode(DIN_PIN0, PIN_MODE::INPUT); //DIN0
    pinMode(DIN_PIN1, PIN_MODE::INPUT); //DIN1
    pinMode(DIN_PIN2, PIN_MODE::INPUT); //DIN2
    pinMode(DIN_PIN3, PIN_MODE::INPUT); //DIN3
    pinMode(DIN_PIN4, PIN_MODE::INPUT); //DIN4
    pinMode(DIN_PIN5, PIN_MODE::INPUT); //DIN5
    pinMode(DIN_PIN6, PIN_MODE::INPUT); //DIN6
    pinMode(DIN_PIN7, PIN_MODE::INPUT); //DIN7
}

/*** the loop procedure (called every event) */
void loop() {
    uint8_t din0, din1, din2, din3, din4, din5, din6, din7;
    uint8_t channel = 0;
    uint8_t sensor_id = 0;

    din0 = (digitalRead(DIN_PIN0) == HIGH) ? 1: 0;
    din1 = (digitalRead(DIN_PIN1) == HIGH) ? 1: 0;
    din2 = (digitalRead(DIN_PIN2) == HIGH) ? 1: 0;
    din3 = (digitalRead(DIN_PIN3) == HIGH) ? 1: 0;
    din4 = (digitalRead(DIN_PIN4) == HIGH) ? 1: 0;
    din5 = (digitalRead(DIN_PIN5) == HIGH) ? 1: 0;
    din6 = (digitalRead(DIN_PIN6) == HIGH) ? 1: 0;
    din7 = (digitalRead(DIN_PIN7) == HIGH) ? 1: 0;

    channel  = din3;
    channel |= din2<<1;
    channel |= din1<<2;
    channel |= din0<<3;

    sensor_id  = din7;
    sensor_id |= din6<<1;
    sensor_id |= din5<<2;
    sensor_id |= din4<<3;

    Serial.print(channel);
    Serial.print("\t");
    Serial.print(sensor_id);
    Serial.print("\t");

    Serial.print(din0);
    Serial.print("\t");
    Serial.print(din1);
    Serial.print("\t");
    Serial.print(din2);
    Serial.print("\t");
    Serial.print(din3);
    Serial.print("\t");
    Serial.print(din4);
    Serial.print("\t");
    Serial.print(din5);
    Serial.print("\t");
    Serial.print(din6);
    Serial.print("\t");
    Serial.println(din7);

    delay(100);
}

