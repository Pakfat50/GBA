#include <TWELITE>
#include "MadgwickAHRS.h"
#include "bmi160.h"
#include "utility.h"

typedef struct{
    uint32_t systime;
    float ax;
    float ay;
    float az;
    float gx;
    float gy;
    float gz;
    float roll;
    float pitch;
    float yaw;
    float roll_raw;
    float pitch_raw;
    float yaw_raw;
}IMU_DATA;

Madgwick imu_filter;
struct bmi160_dev bmi160;
IMU_DATA imu_data;
struct bmi160_sensor_data accel;
struct bmi160_sensor_data gyro;

LOWPASS roll_lowp;
LOWPASS pitch_lowp;

static int8_t imu_init(void);
static int8_t user_i2c_read(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t len);
static int8_t user_i2c_write(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t len);
static void user_delay_ms(uint32_t period);

uint32_t t_system = 0;
uint32_t t_now = 0;
int32_t delay_time = 0;
uint32_t t_transmit_delay = 0;

void setup() {
    Wire.begin(WIRE_CONF::WIRE_50KHZ, false);
    imu_init();
    Serial.println("Start");
    imu_filter.begin(20.0);
}

void loop() {
    
    bmi160_get_sensor_data((BMI160_ACCEL_SEL | BMI160_GYRO_SEL), &accel, &gyro, &bmi160);
    imu_data.systime = t_system;
    
    imu_data.ax = (float)accel.z/16384.0;
    imu_data.ay = (float)accel.y/16384.0;
    imu_data.az = -(float)accel.x/16384.0;
    imu_data.gx = (float)gyro.z /131.0;
    imu_data.gy = (float)gyro.y /131.0;
    imu_data.gz = -(float)gyro.x /131.0;

    imu_filter.updateIMU(imu_data.gx, imu_data.gy, imu_data.gz, imu_data.ax, imu_data.ay, imu_data.az);
    //imu_filter.updateIMU(0, 0, 0, 1, 1, 0.99);

    imu_data.roll_raw = imu_filter.getRoll();
    imu_data.pitch_raw = imu_filter.getPitch();
    imu_data.yaw_raw = imu_filter.getYaw();
    
    
    Serial.print(t_system);
    Serial.print("\t");  
    Serial.print(imu_data.ax);    
    Serial.print("\t");  
    Serial.print(imu_data.ay);    
    Serial.print("\t");  
    Serial.print(imu_data.az); 
    Serial.print("\t");  
    Serial.print(imu_data.gx);    
    Serial.print("\t");  
    Serial.print(imu_data.gy);    
    Serial.print("\t");  
    Serial.print(imu_data.gz);                 
    Serial.print("\t");  
    Serial.print(imu_data.roll_raw);
    Serial.print("\t");    
    Serial.print(imu_data.pitch_raw);
    Serial.print("\t");    
    Serial.print(imu_data.yaw_raw);
    Serial.println();   
    

    t_now = micros();
    uint32_t delta_t = t_now - t_system;
    delay_time = 50000-delta_t;

    if( (delay_time > 0) && (delay_time < 50000)){
        delayMicroseconds(delay_time);
    }
    t_system = micros();
    

}

static int8_t imu_init(void){
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
    bmi160.gyro_cfg.range = BMI160_GYRO_RANGE_250_DPS;
    bmi160.gyro_cfg.bw = BMI160_GYRO_BW_NORMAL_MODE;
    bmi160.gyro_cfg.power = BMI160_GYRO_NORMAL_MODE;
    ret = bmi160_set_sens_conf(&bmi160);
    delay(500);

    return ret;
}

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