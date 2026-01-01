#include "serial_parser.h"
#include <TWELITE>
#include <stdio.h>
#include <stdlib.h>

static void inizializeArray(char* array, size_t array_size);

bool parseMode(uint8_t b_data, uint8_t *mode){
    bool l_res = false;

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
                    l_res = true;

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
            }
            break;
    }

    return l_res;
}


bool parseVal(uint8_t b_data, uint8_t *cls_id, float*val){

    bool l_res = false;

    static MES_VAL s_mes_state =  VAL_HEADER;
    static uint16_t s_mes_pos = 0;
    static uint8_t s_cls_id = 0;
    static char s_payload[MAX_BUF_SIZE] = {0};

    switch (s_mes_state){
        case VAL_HEADER:
            if (b_data == val_header[s_mes_pos]){
                if (s_mes_pos==sizeof(val_header)-1){
                    s_mes_pos = 0;
                    s_mes_state = CLS_ID;
                }
                else{
                    s_mes_pos +=1;
                }
            }
            else{
                s_mes_pos = 0;
                s_cls_id = 0;
                inizializeArray(s_payload, MAX_BUF_SIZE);
            }
            break;

        case CLS_ID:
            s_cls_id = b_data;
            s_mes_pos = 0;
            s_mes_state = CRLF1;
            break;
        
        case CRLF1:
            if (b_data == b_crlf[s_mes_pos]){
                if (s_mes_pos==sizeof(b_crlf)-1){
                    s_mes_pos = 0;
                    s_mes_state = VAL;
                }
                else{
                    s_mes_pos +=1;
                }
            }
            else{
                s_mes_pos = 0;
                s_cls_id = 0;
                inizializeArray(s_payload, MAX_BUF_SIZE);
                s_mes_state = VAL_HEADER;
            }
            break;

        case VAL:
            if((s_mes_pos < MAX_BUF_SIZE)&&(b_data != val_dilimiter)){
                s_payload[s_mes_pos] = (char)b_data;
                s_mes_pos += 1;
            }
            else{
                s_mes_pos = 0;
                s_mes_state = CRLF2;
            }
            break;

        case CRLF2:
            if (b_data == b_crlf[s_mes_pos]){
                if (s_mes_pos==sizeof(b_crlf)-1){
                    *cls_id = s_cls_id;
                    *val = (float)atof(s_payload);
                    l_res = true;

                    s_mes_pos = 0;
                    s_cls_id = 0;
                    inizializeArray(s_payload, MAX_BUF_SIZE);
                    s_mes_state = VAL_HEADER;
                }
                else{
                    s_mes_pos +=1;
                }
            }
            else{
                s_mes_pos = 0;
                s_cls_id = 0;
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