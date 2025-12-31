import processing.serial.*;
import java.nio.ByteBuffer;
Serial port;
int  data;

char mes_disable_nema[]    = {0xBA,0xCE,0x08,0x00,0x06,0x00,0xFF,0x13,0xC0,0x08,0x80,0x25,0x00,0x00,0x87,0x39,0xC6,0x08};
char mes_enable_nema[]     = {0xBA,0xCE,0x08,0x00,0x06,0x00,0xFF,0x33,0xC0,0x08,0x80,0x25,0x00,0x00,0x87,0x59,0xC6,0x08};
char mes_disable_pv[]      = {0xBA,0xCE,0x04,0x00,0x06,0x01,0x01,0x00,0x00,0x00,0x05,0x00,0x06,0x01}; 
char mes_disable_timeutc[] = {0xBA,0xCE,0x04,0x00,0x06,0x01,0x01,0x10,0x00,0x00,0x05,0x10,0x06,0x01}; 
char mes_enable_pv[]       = {0xBA,0xCE,0x04,0x00,0x06,0x01,0x01,0x03,0x01,0x00,0x05,0x03,0x07,0x01};
char mes_enable_timeutc[]  = {0xBA,0xCE,0x04,0x00,0x06,0x01,0x01,0x10,0x01,0x00,0x05,0x10,0x07,0x01};
String str_nema_pcsa04 = "$PCAS04,2C*58\r\n";

public enum MES_STATE{
  HEADER,
  LENGTH,
  CLSID,
  SUBID,
  PAYLOAD,
  CHKSUM_CHK
};

MES_STATE g_mes_state =  MES_STATE.HEADER;
int MAX_BUF_SIZE = 120;
int g_chk_sum = 0;
int g_mes_pos = 0;
int g_mes_len = 0;
byte g_clsid = 0;
byte g_sub_id = 0;
byte g_payload[] = new byte[MAX_BUF_SIZE];
byte g_chksum_byte[] = new byte[4];



void setup() 
{
  port = new Serial(this, Serial.list()[0], 115200);
  size(300, 300);
  background(0, 0, 0);
  
  send_str(str_nema_pcsa04);
  send_mes(mes_disable_nema);
  send_mes(mes_enable_pv);
  send_mes(mes_enable_timeutc);
  //send_mes(mes_enable_nema);
}

void draw() 
{
}

void serialEvent(Serial port)
{  
  byte b_data = 0;
  // シリアルポートからデータを受け取ったら
  while (port.available() >=1) 
  {     
    data = port.read();
    b_data = (byte)data;
    if(casic_parser(b_data) == true){
        println(g_mes_len);
        println(hex(g_clsid,2));
        println(hex(g_sub_id,2));
        for(int i=0; i<g_mes_len; i++){
          print(hex(g_payload[i],2));
          print(" ");
        }
        println();
        print(hex(g_chk_sum,8));
        println();
        
        int ckSum = calc_chksum(g_clsid, g_sub_id, g_mes_len, g_payload);
        print(hex(ckSum,8));
        println();
    }
    
    //print(hex(b_data));
    //print(" ");

  } 
}

void send_mes(char mes[]){
  byte b_mes[] = new byte[mes.length];
  
  for(int i=0; i<mes.length; i++){
    b_mes[i] = (byte)mes[i];
  }
  
  port.write(b_mes);
  
  /*
  //for debug
  for(int i=0; i<b_mes.length; i++){
    print(hex(b_mes[i]));
  }
  println();
  */
}

void send_str(String mes){
  port.write(mes);
}

boolean casic_parser(byte b_data){
  boolean l_res = false;
  switch (g_mes_state){
    case HEADER:
      if ((g_mes_pos==0)&&(b_data == (byte)0xBA)){
        //print(b_data);
        g_mes_pos +=1;
      }
      else if((g_mes_pos==1)&&(b_data == (byte)0xCE)){
        g_mes_pos = 0;
        g_mes_state = MES_STATE.LENGTH;
      }
      else{
        g_chk_sum = 0;
        g_mes_pos = 0;
        g_mes_len = 0;
        g_clsid = 0;
        g_sub_id = 0;
      }
      break;
    case LENGTH:
      if(g_mes_pos == 0){
        g_mes_pos += 1;
        g_mes_len = (int)b_data;
      }
      else{
        g_mes_pos = 0;
        g_mes_len += (int)b_data<<8;
        g_mes_state = MES_STATE.CLSID;
        //println(g_mes_len);
      }
      break;
    case CLSID:
        g_clsid = b_data;
        g_mes_state = MES_STATE.SUBID;
      break;
    case SUBID:
        g_sub_id = b_data;
        g_mes_state = MES_STATE.PAYLOAD;
      break;
    case PAYLOAD:
      if((g_mes_pos < g_mes_len-1)&&(g_mes_pos < MAX_BUF_SIZE-1)){
        g_payload[g_mes_pos] = b_data;
        g_mes_pos += 1;
      }
      else{
        g_payload[g_mes_pos] = b_data;
        g_mes_pos = 0;
        g_mes_state = MES_STATE.CHKSUM_CHK;
      }
      break;
    case CHKSUM_CHK:
      if((g_mes_pos < 3)){
        g_chksum_byte[3-g_mes_pos] = b_data;
        g_mes_pos += 1;
      }
      else{
        g_mes_pos = 0;
        g_chksum_byte[0] = b_data;
        g_chk_sum = ByteBuffer.wrap(g_chksum_byte).getInt();
        g_mes_state = MES_STATE.HEADER;
        l_res = true;
      }
      break;
  }
  
  return l_res;
}

int calc_chksum(byte clsid, byte sub_id, int len, byte[] payload){
  int ckSum = 0;
  ckSum  = ((int)sub_id)<<24;
  ckSum += ((int)clsid)<<16;
  ckSum += len;
  
  for(int i=0; i<(len/4); i++){
    int l_payload = 0;
    byte b_payload[] = {payload[i*4+3], payload[i*4+2], payload[i*4+1], payload[i*4]};
    l_payload = ByteBuffer.wrap(b_payload).getInt();
    //println(hex(l_payload));
    ckSum += l_payload;
  }
  return ckSum;
}
