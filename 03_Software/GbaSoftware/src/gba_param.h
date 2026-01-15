#ifndef GBA_PARAM_H_
#define GBA_PARAM_H_

#include <TWELITE>

#define ANGLE_OFFSET_MAX 360.0
#define ANGLE_OFFSET_MIN -360.0
#define ANGLE_OFFSET_DEFAULT 0.0

#define ROLL_OFFSET_MAX 360.0
#define ROLL_OFFSET_MIN -360.0
#define ROLL_OFFSET_DEFAULT 0.0

#define PITCH_OFFSET_MAX 360.0
#define PITCH_OFFSET_MIN -360.0
#define PITCH_OFFSET_DEFAULT 0.0

#define COFF_ANGLE_MAX 10.0
#define COFF_ANGLE_MIN 0.0
#define COFF_ANGLE_DEFAULT 0.04174

//#define PARAM_NUM_MAX ((uint16_t)EEPROM_ADDR_END-(uint16_t)EEPROM_ADDR_BASE)/4
#define PARAM_NUM_MAX sizeof(GBA_PARAM)/sizeof(float)


#pragma pack(push,1)
typedef struct{
    float angleOffset0;  // Param Num: 0
    float angleOffset1;  // Param Num: 1
    float rollOffset;    // Param Num: 2
    float pitchOffset;   // Param Num: 3
    float coffAngle0;    // Param Num: 4
    float coffAngle1;    // Param Num: 5
}GBA_PARAM;
#pragma pack(pop)

bool get_gba_param(GBA_PARAM *gba_param);
bool set_gba_param(uint16_t param_num, float val);


#endif