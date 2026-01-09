#include <TWELITE>
#include "normal_mode_task.h"
#include "gba_mode.h"
#include "gba_param.h"
#include "gba_twenet.h"
#include "din.h"
#include "mt6701.h"
#include "AHT20.h"
#include "BMP280.h"
#include "bmi160.h"
#include "casic_parser.h"
#include "serial_parser.h"

static int8_t user_i2c_read(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t len);
static int8_t user_i2c_write(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t len);
static void user_delay_ms(uint32_t period);
static void bmi160_init(void);

MT6701 angle0;
MT6701 angle1;
BMP280 bmp280;
struct bmi160_dev bmi160;

void normalModeInit(void){
    Wire.begin(WIRE_CONF::WIRE_50KHZ, false);

    angle0.init(0, &SPI);
    angle1.init(1, &SPI);
    aht20_begin();
    bmp280.init();
    bmi160_init();

    dinInit();


}


GBA_MODE normalModeTask(GBA_PARAM gba_param){

}




// https://smtengkapi.com/engineer-twelite-i2c

static int8_t user_i2c_read(uint8_t dev_id, uint8_t reg_addr, uint8_t *reg_data, uint16_t len)
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

static int8_t user_i2c_write(uint8_t dev_id, uint8_t reg_addr, uint8_t *reg_data, uint16_t len)
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
static void user_delay_ms(uint32_t period) {
    delay(period);
}

static int8_t bmi160_init(void){
    int8_t ret = BMI160_OK;

    // BMI160 センサの初期化.
    bmi160.id = 0x69;  // 0x68 or 0x69
    bmi160.intf = BMI160_I2C_INTF;
    bmi160.read = user_i2c_read;      // [1] (※後述）
    bmi160.write = user_i2c_write;    // [2]
    bmi160.delay_ms = user_delay_ms;  // [3]
    ret = bmi160_init(&bmi160);

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
    delay(500);

    return ret;
}