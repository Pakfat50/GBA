#include <TWELITE>
#include "bmi160.h"
#include "bmi160_defs.h"

/* https://qiita.com/nhiro/items/ec9c4137f0fd81ba78ab */

struct bmi160_dev bmi160;

int8_t user_i2c_read(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t len);
int8_t user_i2c_write(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t len);
void user_delay_ms(uint32_t period);

/*** the setup procedure (called on boot) */
void setup() {

    delay(500);
    Wire.begin(WIRE_CONF::WIRE_50KHZ, false);
    int8_t ret = BMI160_OK;

    // BMI160 センサの初期化.
    bmi160.id = 0x69;  // 0x68 or 0x69
    bmi160.intf = BMI160_I2C_INTF;
    bmi160.read = user_i2c_read;      // [1] (※後述）
    bmi160.write = user_i2c_write;    // [2]
    bmi160.delay_ms = user_delay_ms;  // [3]
    ret = bmi160_init(&bmi160);
    Serial.print("bmi160_init: ");
    Serial.println(ret);

    // Accel の設定.
    bmi160.accel_cfg.odr = BMI160_ACCEL_ODR_100HZ;
    bmi160.accel_cfg.range = BMI160_ACCEL_RANGE_2G;
    bmi160.accel_cfg.bw = BMI160_ACCEL_BW_NORMAL_AVG4;
    bmi160.accel_cfg.power = BMI160_ACCEL_NORMAL_MODE;
    // Gyro の設定.
    bmi160.gyro_cfg.odr = BMI160_GYRO_ODR_100HZ;
    bmi160.gyro_cfg.range = BMI160_GYRO_RANGE_2000_DPS;
    bmi160.gyro_cfg.bw = BMI160_GYRO_BW_NORMAL_MODE;
    bmi160.gyro_cfg.power = BMI160_GYRO_NORMAL_MODE;
    ret = bmi160_set_sens_conf(&bmi160);
    Serial.print("bmi160_set_sens_conf: ");
    Serial.println(ret);

    Serial.println("========================================");
    Serial.println("accl.x\taccl.y\taccl.z\tgyro.x\tgyro.y\tgyro.z");
    delay(500);
}

/*** the loop procedure (called every event) */
void loop() {
    struct bmi160_sensor_data accel;
    struct bmi160_sensor_data gyro;

    // Accel, Gyro の取得.
    bmi160_get_sensor_data(
    (BMI160_ACCEL_SEL | BMI160_GYRO_SEL),
    &accel, &gyro, &bmi160);

    // シリアルへの出力.
    Serial.print(accel.x);
    Serial.print("\t");
    Serial.print(accel.y);
    Serial.print("\t");
    Serial.print(accel.z);
    Serial.print("\t");
    Serial.print(gyro.x);
    Serial.print("\t");
    Serial.print(gyro.y);
    Serial.print("\t");
    Serial.print(gyro.z);
    Serial.println();

    delay(10);

}

// https://smtengkapi.com/engineer-twelite-i2c

int8_t user_i2c_read(uint8_t dev_id, uint8_t reg_addr, uint8_t *reg_data, uint16_t len)
{   
    int8_t rslt = 0;
    if(auto&& wrt = Wire.get_writer(dev_id)){ //読み込みたいアドレスを指定
        wrt << reg_addr;
    }

    if(auto&& rdr = Wire.get_reader(dev_id,(uint8_t)len)){  //len分データをread
        for(uint16_t i = 0; i < len; i++){
            *reg_data = rdr();
            reg_data++; //ポインターを更新
        }
    }
    return rslt;
}

int8_t user_i2c_write(uint8_t dev_id, uint8_t reg_addr, uint8_t *reg_data, uint16_t len)
{   
    int8_t rslt = 0; 
    if(auto&& wrt = Wire.get_writer(dev_id)){ //読み込みたいアドレスを指定
        wrt << reg_addr;
        for(uint16_t i = 0; i < len; i++){ //len分データをwrite
            wrt << reg_data[i]; //*reg_data;でもよいがポインタの更新が必要
        }
    }
    return rslt;
}

// [3]
void user_delay_ms(uint32_t period) {
    delay(period);
}