#ifndef MT6701_H_
#define MT6701_H_

#include <TWELITE>

#define SPICLOCK 8000000
#define SPI_SETTING SPISettings(SPICLOCK, SPI_CONF::MSBFIRST, SPI_CONF::SPI_MODE1)

typedef enum{
    NORMAL,
    TOO_STRONG,
    TOO_WEAK,
    NC
}MAG_STRENGTH;

typedef enum{
    NOT_DETECT,
    DETECT
}PUSH_BOTTON;

typedef enum{
    TRACKING,
    LOSS
}TRACK;


class MT6701{
    public:
        void init(uint8_t slave_select, mwx::periph_spi* spi);
        void getAngle(float* angle);
        void getStatus(MAG_STRENGTH* mag_strength, PUSH_BOTTON* push_botton, TRACK* track);
    private:
        uint8_t _slave_select;
        mwx::periph_spi* _spi;
        void readData(uint8_t b_data[3]);
};


#endif