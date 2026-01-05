#include "serial_parser.h"
#include <TWELITE>
#include <stdio.h>
#include <stdlib.h>

static void inizializeArray(char* array, size_t array_size);

SERIAL_ERR parseMode(uint8_t b_data, uint8_t *mode){
    SERIAL_ERR l_res = PARSING;

    static MES_MODE s_mes_state =  MODE_HEADER;
    static uint16_t s_mes_pos = 0;
    static uint8_t s_mode = 0;

    switch (s_mes_state){
        case MODE_HEADER:
            if (b_data == mode_header[s_mes_pos]){
                if (s_mes_pos==sizeof(mode_header)-1){
                    s_mes_pos = 0;
                    s_mes_state = MODE;
                }
                else{
                    s_mes_pos +=1;
                }
            }
            else{
                s_mes_pos = 0;
                s_mode = 0;
            }
            break;

        case MODE:
            s_mode = b_data;
            s_mes_pos = 0;
            s_mes_state = CRLF;
            break;

        case CRLF:
            if (b_data == b_crlf[s_mes_pos]){
                if (s_mes_pos==sizeof(b_crlf)-1){
                    *mode = s_mode;
                    l_res = GET_VALUE;

                    s_mes_pos = 0;
                    s_mode = 0;
                    s_mes_state = MODE_HEADER;
                }
                else{
                    s_mes_pos +=1;
                }
            }
            else{
                s_mes_pos = 0;
                s_mode = 0;
                s_mes_state = MODE_HEADER;
                l_res = ERR_CRLF;
            }
            break;
    }

    return l_res;
}


SERIAL_ERR parseVal(uint8_t b_data, uint16_t *param_num, float*val){
    SERIAL_ERR l_res = PARSING;

    static MES_VAL s_mes_state =  VAL_HEADER;
    static uint16_t s_mes_pos = 0;
    static char s_param_num[MAX_BUF_SIZE] = {0};
    static char s_payload[MAX_BUF_SIZE] = {0};

    switch (s_mes_state){
        case VAL_HEADER:
            if (b_data == val_header[s_mes_pos]){
                if (s_mes_pos==sizeof(val_header)-1){
                    s_mes_pos = 0;
                    s_mes_state = PARAM_NUM;
                }
                else{
                    s_mes_pos +=1;
                }
            }
            else{
                s_mes_pos = 0;
                inizializeArray(s_param_num, MAX_BUF_SIZE);
                inizializeArray(s_payload, MAX_BUF_SIZE);
            }
            break;

        case PARAM_NUM:
            // LRのみの場合を想定し、b_data != b_crlf[1]を追加
            if((s_mes_pos < MAX_BUF_SIZE)&&(b_data != b_crlf[0])&&(b_data != b_crlf[1])){
                s_param_num[s_mes_pos] = (char)b_data;
                s_mes_pos += 1;
            }
            else{
                s_mes_pos = 0;
                s_mes_state = CRLF1;
            }
            break;

        case CRLF1:
            if (b_data == b_crlf[1]){
                s_mes_pos = 0;
                s_mes_state = VAL;
            }
            else{
                l_res = ERR_CRLF;
                s_mes_pos = 0;
                inizializeArray(s_param_num, MAX_BUF_SIZE);
                inizializeArray(s_payload, MAX_BUF_SIZE);
                s_mes_state = VAL_HEADER;
            }
            break;

        case VAL:
            // LRのみの場合を想定し、b_data != b_crlf[1]を追加
            if((s_mes_pos < MAX_BUF_SIZE)&&(b_data != b_crlf[0])&&(b_data != b_crlf[1])){
                s_payload[s_mes_pos] = (char)b_data;
                s_mes_pos += 1;
            }
            else{
                s_mes_pos = 0;
                s_mes_state = CRLF2;
            }
            break;

        case CRLF2:
            if (b_data == b_crlf[1]){
                char* param_num_end;
                char* val_end;
                uint16_t  temp_param_num = (uint16_t)strtoul(s_param_num, &param_num_end, 10);
                float temp_val = (float)strtod(s_payload, &val_end);

                if((*param_num_end=='\0') && (*val_end=='\0') && (temp_param_num < PARAM_NUM_MAX)){
                    l_res = GET_VALUE;
                    *param_num = temp_param_num;
                    *val = temp_val;
                }
                else{
                    if(*param_num_end !='\0'){
                        l_res = ERR_INVALID_ID;
                    }
                    if(*val_end !='\0'){
                        l_res = ERR_INVALID_VAL;
                    }
                    if(temp_param_num >= PARAM_NUM_MAX){
                        l_res = ERR_OVER_RANGE;
                    }
                }

                s_mes_pos = 0;
                inizializeArray(s_param_num, MAX_BUF_SIZE);
                inizializeArray(s_payload, MAX_BUF_SIZE);
                s_mes_state = VAL_HEADER;
            }
            else{
                l_res = ERR_CRLF;
                s_mes_pos = 0;
                inizializeArray(s_param_num, MAX_BUF_SIZE);
                inizializeArray(s_payload, MAX_BUF_SIZE);
                s_mes_state = VAL_HEADER;
            }
            break;
    } 
    
    return l_res;
}

static void inizializeArray(char* array, size_t array_size){
    for(size_t i=0; i<array_size; i++){
        array[i] = 0;
    }
}