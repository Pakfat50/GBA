#include "mt6701.h"
#include <TWELITE>

void MT6701::init(uint8_t slave_select, mwx::periph_spi* spi){
    _slave_select = slave_select;
    _spi = spi;
}

void MT6701::getAngle(float* angle){
    uint8_t b_data[3] = {0};
    int i_angle = 0;

    readData(b_data);

    i_angle += ((int)b_data[0])<<6; //top 8 bits of 14bits
    i_angle += ((int)b_data[1])>>2; //bottom 6 bits of 14 bits

    *angle = float(i_angle)*360.0/16384.0;
}

void MT6701::getStatus(MAG_STRENGTH* mag_strength, PUSH_BOTTON* push_botton, TRACK* track){
    uint8_t b_data[3] = {0};
    uint8_t b_mag_strength, b_push_botton, b_track;

    readData(b_data);

    b_track = (b_data[1]>>1) & 0x01;
    b_push_botton = (b_data[1]) & 0x01;
    b_mag_strength = (b_data[2]>>6) & 0x03;

    if(b_track == 1){
        *track = MAG_LOSS;
    }
    else{
        *track = MAG_TRACKING;
    }

    if(b_push_botton == 1){
        *push_botton = MAG_DETECT;
    }
    else{
        *push_botton = MAG_NOT_DETECT;
    }

    switch (b_mag_strength){
        case 0:
            *mag_strength = MAG_NORMAL;
            break;
        case 1:
            *mag_strength = MAG_TOO_STRONG;
            break;
        case 2:
            *mag_strength = MAG_TOO_WEAK;
            break;
        default:
            *mag_strength = MAG_NC;
            break;
    }
}

void MT6701::readData(uint8_t b_data[3]){

    _spi->begin(_slave_select, SPI_SETTING);
    _spi->beginTransaction();
    b_data[0] = _spi->transfer(0x00);
    b_data[1] = _spi->transfer(0x00);
    b_data[2] = _spi->transfer(0x00);
    _spi->endTransaction();

}
