#ifndef MT6701_H_
#define MT6701_H_

#include <TWELITE>

#define SPICLOCK 8000000
#define SPI_SETTING SPISettings(SPICLOCK, SPI_CONF::MSBFIRST, SPI_CONF::SPI_MODE1)

typedef enum{
    MAG_NORMAL,
    MAG_TOO_STRONG,
    MAG_TOO_WEAK,
    MAG_NC
}MAG_STRENGTH;

typedef enum{
    MAG_NOT_DETECT,
    MAG_DETECT
}PUSH_BOTTON;

typedef enum{
    MAG_TRACKING,
    MAG_LOSS
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