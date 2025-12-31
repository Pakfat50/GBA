#include <TWELITE>


/*** the setup procedure (called on boot) */
void setup() {
    SPI.begin(1, SPISettings(1000000, SPI_CONF::MSBFIRST, SPI_CONF::SPI_MODE1));
}

/*** the loop procedure (called every event) */
void loop() {
    int angle=0;

    SPI.beginTransaction();
    char data1 = SPI.transfer(0x00);
    char data2 = SPI.transfer(0x00);
    char data3 = SPI.transfer(0x00);
    SPI.endTransaction();

    angle += (int)data1<<6; //top 8 bits of 14bits
    angle += (int)data2>>2; //bottom 6 bits of 14 bits

    Serial.print(angle);
    Serial.print(", ");
    Serial.print(float(angle)*360.0/16384.0);
    Serial.print(", ");
    Serial.print(data1);
    Serial.print(", ");
    Serial.print(data2);
    Serial.print(", ");
    Serial.print(data3);
    Serial.println();

    delay(100);

}