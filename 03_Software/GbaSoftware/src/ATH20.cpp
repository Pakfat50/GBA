#include "AHT20.h"
#include <TWELITE>

static bool aht20_isBusy(void);

void aht20_begin()
{   
    if(auto&& wrt = Wire.get_writer(0x38)){ //読み込みたいアドレスを指定
        wrt << 0xBE;
    }
}

bool aht20_startSensor()
{   
    if(aht20_isBusy() == true){
        return false;
    }

    if(auto&& wrt = Wire.get_writer(0x38)){ //読み込みたいアドレスを指定
        wrt << 0xac;
        wrt << 0x33;
        wrt << 0x00;
    }
    return true;
    
}

static bool aht20_isBusy(void){
    bool ret_val = false;

    if(auto&& rdr = Wire.get_reader(0x38, 1)){  //1byte分データをreads
        unsigned char c = rdr();
        if( (c&0x80) == 1){
            ret_val = true;      // busy
        }
    }

    return ret_val;
}

bool aht20_getSensor(float *h, float *t)
{
    unsigned char str[6] ={0,};
    int index = 0;
    
    if(auto&& rdr = Wire.get_reader(0x38, 6)){  //6byte分データをread
        for(uint16_t i = 0; i < 6; i++){
            str[index] = rdr();
            index++;
        }
    }

    if(index == 0 ) {
        return false;
    } 

    if((str[0] & 0x80) == 1){
        return false;
    }


    unsigned long __humi = 0;
    unsigned long __temp = 0;

    __humi = str[1];
    __humi <<= 8;
    __humi += str[2];
    __humi <<= 4;
    __humi += str[3] >> 4;

    *h = (float)__humi/1048576.0;

    __temp = str[3]&0x0f;
    __temp <<=8;
    __temp += str[4];
    __temp <<=8;
    __temp += str[5];

    *t = (float)__temp/1048576.0*200.0-50.0;

    return true;

}

bool aht20_getTemperature(float *t)
{
    float __t, __h;
    
    int ret = aht20_getSensor(&__h, &__t);
    if(0 == ret)return 0;
    
    *t = __t;
    return 1;
}

bool aht20_getHumidity(float *h)
{
    float __t, __h;
    
    int ret = aht20_getSensor(&__h, &__t);
    if(0 == ret)return 0;
    
    *h = __h;
    return 1;
}


