#include <TWELITE>
#include <math.h>
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
#include "calc_wind_speed.h"
#include "logger.h"
#include "utility.h"

static int8_t user_i2c_read(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t len);
static int8_t user_i2c_write(uint8_t dev_addr, uint8_t reg_addr, uint8_t* data, uint16_t len);
static void user_delay_ms(uint32_t period);
static int8_t imu_init(void);
static void gps_init(void);
static void update_gps(void);

#ifdef DEBUG_NORMAL_MODE
static void print_debug_messeage(uint8_t mode);
#define DEBUG_PLOT_MES
#endif 

GBA_DATA gba_data;
ANGLE_DATA angle_data;
IMU_DATA imu_data;
ENV_DATA env_data;
WIND_DATA wind_data;
STATUS_DATA status_data;
NAV_PV nav_pv;
NAV_TIMEUTC nav_timeutc;
struct bmi160_sensor_data accel;
struct bmi160_sensor_data gyro;

uint8_t b_gba_data[sizeof(GBA_DATA)] = {0};
uint8_t b_angle_data[sizeof(ANGLE_DATA)] = {0};
uint8_t b_imu_data[sizeof(IMU_DATA)] = {0};
uint8_t b_env_data[sizeof(ENV_DATA)] = {0};
uint8_t b_wind_data[sizeof(WIND_DATA)] = {0};
uint8_t b_status_data[sizeof(STATUS_DATA)] = {0};
uint8_t b_nav_pv[sizeof(NAV_PV)] = {0};
uint8_t b_nav_timeutc[sizeof(NAV_TIMEUTC)] = {0};
uint8_t b_status[sizeof(STATUS_DATA)] = {0};

MT6701 angleSensor0;
MT6701 angleSensor1;
BMP280 bmp280;
struct bmi160_dev bmi160;
MOVING_AVERAGE windAverage0;
MOVING_AVERAGE windAverage1;
MIN_MAX windMax0;
MIN_MAX windMax1;
LOWPASS axLowp;
LOWPASS ayLowp;
LOWPASS azLowp;
LOWPASS rollLowp;
LOWPASS pitchLowp;


uint32_t high_rate_cnt = 0;
uint32_t mid_rate_cnt = 0;
uint32_t t_system = 0;
uint32_t t_old = 0;
uint32_t t_now = 0;
uint32_t t_pps = 0;
int32_t delay_time = 0;
uint32_t t_transmit_delay = 0;

#ifdef DEBUG_NORMAL_MODE
uint32_t high_rate_t_max = 0;
uint32_t mid_rate_t_max = 0;
uint32_t high_rate_max_cnt = 0;
uint32_t mid_rate_max_cnt = 0;
#define DELIMITER "\t"
#endif

bool do_transmit = false;


void normalModeInit(void){
    Wire.begin(WIRE_CONF::WIRE_50KHZ, false);

    angleSensor0.init(0, &SPI);
    angleSensor1.init(1, &SPI);
    aht20_begin();
    bmp280.init();
    imu_init();
    gps_init();
    dinInit();
    uint8_t channel = getChannel();
    uint8_t sensor_id = getSensorId();

    gbaTwenetInit(sensor_id, channel, &the_twelite);

    //PPS割り込み処理

    windAverage0.init(NUM_WIND_AVERAGE, 0.0);
    windAverage1.init(NUM_WIND_AVERAGE, 0.0);
    windMax0.init(0.0);
    windMax1.init(0.0);
    axLowp.init(0.1, 0);
    ayLowp.init(0.1, 0);
    azLowp.init(0.1, 1);
    rollLowp.init(0.1, 0);
    pitchLowp.init(0.1, 0);

    t_transmit_delay = TRANSMIT_INT*(uint32_t)sensor_id;

    Serial.print("Channel = ");
    Serial.print(channel);
    Serial.print(" Sensor ID = ");
    Serial.println(sensor_id);
}


GBA_MODE normalModeTask(GBA_PARAM gba_param){
    bool check_mode = false;
    uint8_t b_mode;
    GBA_MODE ret_mode = NORMAL;
    uint32_t t_transmit_delta = 0;
    float l_temperature, l_humidity;
    static float l_ax = 0.0;
    static float l_ay = 0.0;
    static float l_az = 1.0;

    angle_data.systime = t_system;
    angleSensor0.getAngle(&angle_data.angle0_raw);
    angleSensor1.getAngle(&angle_data.angle1_raw);

    angle_data.angle0 = angle_data.angle0_raw - gba_param.angleOffset0 - imu_data.pitch;
    angle_data.angle1 = angle_data.angle1_raw - gba_param.angleOffset1 - imu_data.roll;

    angle_data.angle0_average = windAverage0.get(angle_data.angle0);
    angle_data.angle1_average = windAverage1.get(angle_data.angle1);
    windMax0.set(angle_data.angle0);
    windMax1.set(angle_data.angle1);

    memcpy(b_angle_data, &angle_data, sizeof(ANGLE_DATA));
#ifndef DEBUG_NORMAL_MODE
    send_data(GBA_CLASS_ID, ANGLE_ID, (uint16_t)sizeof(ANGLE_DATA), b_angle_data, &Serial);
#endif
    update_gps();

    switch (high_rate_cnt)
    {
    case 0:
        bmi160_get_sensor_data((BMI160_ACCEL_SEL | BMI160_GYRO_SEL), &accel, &gyro, &bmi160);
        imu_data.systime = t_system;
        imu_data.ax = accel.z/16384.0;
        imu_data.ay = accel.y/16384.0;
        imu_data.az = -accel.x/16384.0;
        imu_data.gx = gyro.z /131.0 * PI / 180.0;
        imu_data.gy = gyro.y /131.0 * PI / 180.0;
        imu_data.gz = -gyro.x /131.0 * PI / 180.0;
        break;

    case 1:
        l_ax = axLowp.get(imu_data.ax);
        l_ay = ayLowp.get(imu_data.ay);
        l_az = azLowp.get(imu_data.az);
        imu_data.roll_raw = atan2f(l_ax, sqrt(l_ay*l_ay + l_az*l_az)) * 180 / 3.141592;
        imu_data.pitch_raw = atan2f(l_ay, sqrt(l_ax*l_ax + l_az*l_az)) * 180 / 3.141592;           
        break;    

    case 2:
        imu_data.roll = rollLowp.get(imu_data.roll_raw)  - gba_param.rollOffset;
        imu_data.pitch = pitchLowp.get(imu_data.pitch_raw)  - gba_param.pitchOffset;
        break;   

    case 3:
        t_transmit_delta = millis() - t_pps;

        if(t_transmit_delta > TIME_PPS_MAX_WAIT){
            t_pps = millis();
            do_transmit = true;
        }

        if((t_transmit_delta >= t_transmit_delay) && (do_transmit == true)){
            gba_data.year = nav_timeutc.year;
            gba_data.month = nav_timeutc.month;
            gba_data.day = nav_timeutc.day;
            gba_data.hour = nav_timeutc.hour;
            gba_data.min = nav_timeutc.min;
            gba_data.sec = nav_timeutc.sec;
            gba_data.ms = nav_timeutc.ms;
            gba_data.averageWindSpeedE = wind_data.average_east;
            gba_data.averagewindSpeedN = wind_data.average_north;
            gba_data.gustWindSpeedE = wind_data.gust_east;
            gba_data.gustWindSpeedN = wind_data.gust_north;
            gba_data.lon = nav_pv.lon;
            gba_data.lat = nav_pv.lat;
            gba_data.numSV = nav_pv.numSV;
            gba_data.temperature = env_data.temperature;
            gba_data.humidity = env_data.humidity;
            gba_data.pressure = env_data.pressure;

            gbaTwenetTransmit(&gba_data, TRANSMIT_ADDR, &the_twelite);

            windMax0.reset(0.0);
            windMax1.reset(0.0);
            do_transmit = false;
        }else{
            memcpy(b_imu_data, &imu_data, sizeof(IMU_DATA));
#ifndef DEBUG_NORMAL_MODE            
            send_data(GBA_CLASS_ID, IMU_ID, (uint16_t)sizeof(IMU_DATA), b_imu_data, &Serial);
#endif
        }

        break;   

    case 4:
        switch (mid_rate_cnt)
        {
        case 0:
            env_data.systime = t_system;
            if(aht20_getSensor(&l_humidity, &l_temperature)){
                env_data.humidity = l_humidity*100.0;
                env_data.temperature = l_temperature;
            }
            aht20_startSensor();
            break;
        
        case 1:
            env_data.temperature_bmp280 = bmp280.getTemperature();
            env_data.pressure = bmp280.getPressure()/100.0;
            break;            

        case 2:
            angleSensor0.getStatus(&status_data.mag_strength0, &status_data.push_botton0, &status_data.track0);
            angleSensor1.getStatus(&status_data.mag_strength1, &status_data.push_botton1, &status_data.track1);
            break;   

        case 3:
            wind_data.systime = t_system;
            wind_data.gust_east = calcWindSpeed(windMax0.getMaxNorm(), gba_param.coffAngle0);
            wind_data.gust_north = calcWindSpeed(windMax1.getMaxNorm(), gba_param.coffAngle1);
            break;   

        case 4:
            wind_data.average_east = calcWindSpeed(angle_data.angle0_average, gba_param.coffAngle0);
            wind_data.average_north = calcWindSpeed(angle_data.angle1_average, gba_param.coffAngle1);            
            break;   

        case 5:
            memcpy(b_nav_pv, &nav_pv, sizeof(NAV_PV));
#ifndef DEBUG_NORMAL_MODE
            send_data(GPS_NAV_CLSID, GPS_NAVSV_ID, (uint16_t)sizeof(NAV_PV), b_nav_pv, &Serial);
#else
            print_debug_messeage(0);
#endif 
            break;   

        case 6:
            memcpy(b_nav_timeutc, &nav_timeutc, sizeof(NAV_TIMEUTC));
#ifndef DEBUG_NORMAL_MODE
            send_data(GPS_NAV_CLSID, GPS_NAVTIMEUTC_ID, (uint16_t)sizeof(NAV_TIMEUTC), b_nav_timeutc, &Serial);
#else
            print_debug_messeage(1);
#endif
            break;   

        case 7:
            memcpy(b_env_data, &env_data, sizeof(ENV_DATA));
#ifndef DEBUG_NORMAL_MODE
            send_data(GBA_CLASS_ID, ENV_ID, (uint16_t)sizeof(ENV_DATA), b_env_data, &Serial);
#else
            print_debug_messeage(2);
#endif
            break; 
            
        case 8:
            memcpy(b_wind_data, &wind_data, sizeof(WIND_DATA));
#ifndef DEBUG_NORMAL_MODE
            send_data(GBA_CLASS_ID, WIND_ID, (uint16_t)sizeof(WIND_DATA), b_wind_data, &Serial); 
#else
            print_debug_messeage(3);            
#endif
            break; 
            
        case 9:
            memcpy(b_status_data, &status_data, sizeof(STATUS_DATA));
#ifndef DEBUG_NORMAL_MODE
            send_data(GBA_CLASS_ID, STATUS_ID, (uint16_t)sizeof(STATUS_DATA), b_status_data, &Serial);  
#else
            print_debug_messeage(4);            
#endif 
            break;     

        default:
            break;
        }

        mid_rate_cnt += 1;
        if (mid_rate_cnt >= 10){
            mid_rate_cnt = 0;
        }

        break;   

    default:
        break;
    }

    high_rate_cnt += 1;
    if (high_rate_cnt >= 5){
        high_rate_cnt = 0;
    }

    t_now = micros();
    uint32_t delta_t = t_now - t_old;
    delay_time = TIME_INTERVAL_US -delta_t -DELAY_TIME_AJUST;

    if( (delay_time > 0) && (delay_time < TIME_INTERVAL_US)){
        delayMicroseconds(delay_time);
    }
    t_old = micros();
    t_system = millis();

    while(Serial.available()) {
        auto c = Serial.read();
        if(parseMode(c, &b_mode) == GET_VALUE){
            check_mode = true;
        }
    }

    if(check_mode == true){
        ret_mode = mode_checker(b_mode, NORMAL);
    }

#ifdef DEBUG_NORMAL_MODE
    if((delta_t > high_rate_t_max) && (delta_t < DELTA_T_MAX)){
        high_rate_t_max = delta_t;
        high_rate_max_cnt = high_rate_cnt;
        
        if(high_rate_max_cnt != 0){
            high_rate_max_cnt -= 1;
        }
    }
    if((delta_t > mid_rate_t_max) && (delta_t < DELTA_T_MAX)){
        if (high_rate_t_max == 4){
            mid_rate_t_max = delta_t;
            mid_rate_max_cnt = mid_rate_cnt;

            if(mid_rate_max_cnt != 0){
                mid_rate_max_cnt -= 1;
            }
        }
    }

#endif

    return ret_mode;
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

static void gps_init(void){
    gpsSerialInit(&Serial1);
    sendStrMes(str_nema_pcsa04, sizeof(str_nema_pcsa04), &Serial1);
    delay(100);
    sendByteMes(mes_disable_nema, sizeof(mes_disable_nema), &Serial1);
    delay(100);
    sendByteMes(mes_enable_pv, sizeof(mes_enable_pv), &Serial1);
    delay(100);
    sendByteMes(mes_enable_timeutc, sizeof(mes_enable_timeutc), &Serial1);
    delay(100);
}

static void update_gps(void){
    uint32_t chk_sum = 0;
    uint16_t mes_len = 0;
    uint8_t clsid = 0;
    uint8_t sub_id = 0;
    uint8_t gps_payload[GPS_MAX_BUF_SIZE] = {0};

    while(Serial1.available()) {
        auto c_data = Serial1.read();
        uint8_t b_data = (uint8_t)c_data;

        if(casicParser(b_data, &clsid, &sub_id, &mes_len, &chk_sum, gps_payload)==true){
            uint32_t calc_check_sum = calcChksum(clsid, sub_id, mes_len, gps_payload);

            if(chk_sum == calc_check_sum){
                if((clsid==GPS_NAV_CLSID)&&(sub_id==GPS_NAVSV_ID)&&(mes_len==sizeof(nav_pv))){
                    setNavPV(&nav_pv, gps_payload);
                }

                if((clsid==GPS_NAV_CLSID)&&(sub_id==GPS_NAVTIMEUTC_ID)&&(mes_len==sizeof(nav_timeutc))){
                    setNavTimeUtc(&nav_timeutc, gps_payload);
                }
            }
        }
    }
}

#ifdef DEBUG_NORMAL_MODE
static void print_debug_messeage(uint8_t mode){
#ifdef DEBUG_PLOT_MES
    switch (mode)
    {
    case 0:
        Serial.print(t_system);
        Serial.print(DELIMITER);
        Serial.print(high_rate_t_max);
        Serial.print(DELIMITER);   
        Serial.print(high_rate_max_cnt);
        Serial.print(DELIMITER);         
        Serial.print(mid_rate_t_max);
        Serial.print(DELIMITER);   
        Serial.print(mid_rate_max_cnt);
        Serial.print(DELIMITER);           
        break;

    case 1:
        Serial.print(angle_data.angle0);
        Serial.print(DELIMITER);
        Serial.print(angle_data.angle1);
        Serial.print(DELIMITER);            
        break;   

    case 2:
        /*
        Serial.print(imu_data.ax);
        Serial.print(DELIMITER);
        Serial.print(imu_data.ay);
        Serial.print(DELIMITER);    
        Serial.print(imu_data.az);
        Serial.print(DELIMITER);   
        Serial.print(imu_data.gx);
        Serial.print(DELIMITER);
        Serial.print(imu_data.gy);
        Serial.print(DELIMITER);    
        Serial.print(imu_data.gz);
        Serial.print(DELIMITER);  
        */
        Serial.print(imu_data.roll);
        Serial.print(DELIMITER);    
        Serial.print(imu_data.pitch);
        Serial.print(DELIMITER);                         
        break;  

    case 3:
        Serial.print(env_data.temperature);
        Serial.print(DELIMITER);
        Serial.print(env_data.temperature_bmp280);
        Serial.print(DELIMITER);       
        Serial.print(env_data.humidity);
        Serial.print(DELIMITER);              
        Serial.print(env_data.pressure);
        Serial.print(DELIMITER);          
        break; 

    case 4:
        Serial.print(wind_data.average_east);
        Serial.print(DELIMITER);
        Serial.print(wind_data.average_north);
        Serial.print(DELIMITER);       
        Serial.print(wind_data.gust_east);
        Serial.print(DELIMITER);              
        Serial.print(wind_data.gust_north);
        Serial.println(DELIMITER);             
        break;   

    default:
        break;
    }
#endif
}
#endif