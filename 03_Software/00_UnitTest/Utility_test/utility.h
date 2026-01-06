#ifndef UTILITY_H_
#define UTILITY_H_

#include <TWELITE>

class LOWPASS {
    public:
        void init(float k, float x0);
        float get(float x);
    private:
        float _k;
        float _zx;
};

class MOVING_AVERAGE {
    public:
        void init(uint32_t average_num, float x0);
        float get(float x);
    private:
        uint32_t _num;
        uint32_t _average_num;
        float _coff;
        float _zx;
        float _ret_val;
};


#endif