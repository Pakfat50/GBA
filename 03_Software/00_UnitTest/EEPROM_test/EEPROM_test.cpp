#include <TWELITE>

#define ADDR_VAL1 1024
#define ADDR_VAL2 1028

float val1 = 23.4;
float val2 = 1023.4;

typedef union {
  uint8_t U1[4];
  float R4;
}U1_R4;

void eeprom_float_wite(uint16_t addr, float val);
float eeprom_float_read(uint16_t addr);

/*** the setup procedure (called on boot) */
void setup() {
    //eeprom_float_wite(ADDR_VAL1, val1);
    //eeprom_float_wite(ADDR_VAL2, val2);
}

/*** the loop procedure (called every event) */
void loop() {
    float val1_read;
    float val2_read;

    val1_read = eeprom_float_read(ADDR_VAL1);
    val2_read = eeprom_float_read(ADDR_VAL2);

    Serial.print(val1_read);
    Serial.print("\t");
    Serial.println(val2_read);

    delay(1000);
}

void eeprom_float_wite(uint16_t addr, float val){
    U1_R4 u1_r4;
    u1_r4.R4 = val;
    for(uint16_t i=0; i<4; i++){
        EEPROM.update(addr+i, u1_r4.U1[i]);
    }
}

float eeprom_float_read(uint16_t addr){
    U1_R4 u1_r4;
    for(uint16_t i=0; i<4; i++){
        u1_r4.U1[i] = EEPROM.read(addr+i);
    }
    return u1_r4.R4;
}