#include <TinyGPSPlus.h>
 
TinyGPSPlus gps;

//#define DEBUG
#define GPS_ENABLE


constexpr float A_INTER = -0.0803317;
constexpr float B_INTER = 120.248;

constexpr float A_OUTER = -0.0835924;
constexpr float B_OUTER = 203.251;

constexpr float INTER_OFFSET = 0.882998944;
constexpr float OUTER_OFFSET = 5.417771911;

constexpr int   T_INT  = 10000; //[usec], 10mssec :100Hz
constexpr int   OVERHEAD = 16; //[usec]

String delimiter = "\t";

void setup() {
  Serial.begin(115200);
  Serial2.begin(9600,SERIAL_8N1,16,17);
}

float get_inter_angle(uint32_t val)
{
  return A_INTER*float(val) + B_INTER;
}


float get_outer_angle(uint32_t val)
{
  return A_OUTER*float(val) + B_OUTER;
}

void loop()
{ 
  static uint32_t T_NOW = 0;
  static uint32_t T_OLD = 0;
  static uint32_t DT = 0;

  T_OLD = micros();
  uint32_t sys_time = T_OLD; 
  
  #ifdef DEBUG
  while (Serial2.available() > 0) {   
    char res = Serial2.read();
    Serial.print(res);
  }

  while (Serial.available() > 0) {   
    char snd = Serial.read();
    Serial2.print(snd);
  }  
  #else

  uint32_t sensorValue1 = analogRead(26);
  uint32_t sensorValue2 = analogRead(27);
  // print out the value you read:

  float outer_angle = get_outer_angle(sensorValue1) - OUTER_OFFSET;
  float inter_angle = get_inter_angle(sensorValue2) - INTER_OFFSET;


  while (Serial2.available() > 0) {
    gps.encode(Serial2.read());
  }
  
  Serial.print(sys_time);
  Serial.print(delimiter);
  Serial.print(sensorValue1);
  Serial.print(delimiter);
  Serial.print(sensorValue2);
  Serial.print(delimiter);
  Serial.print(outer_angle);
  Serial.print(delimiter);
  Serial.print(inter_angle);
  Serial.print(delimiter);

  #ifdef GPS_ENABLE
  displayInfo();
  #else
  // do notheing;
  #endif

  
  Serial.println();
  #endif

  T_NOW = micros();
  DT = T_NOW - T_OLD;
    
  if (((T_INT - DT - OVERHEAD) > 0 )&&(DT > 0)) { //DT>0を追加
      delayMicroseconds(T_INT - DT - OVERHEAD);
  }


}

void displayInfo() {
  bool res_flag = false;

  if (gps.location.isUpdated()) {
    Serial.print(F("Location:")); 
    Serial.print(F("lat:"));
    Serial.print(gps.location.lat(), 6);
    Serial.print(F("lon:"));
    Serial.print(gps.location.lng(), 6);
    Serial.print(";;");
    res_flag = true;
  } 

  if (gps.date.isUpdated()) {
    Serial.print(F("Date:"));
    Serial.print(gps.date.year());
    Serial.print(F("/"));
    Serial.print(gps.date.month());
    Serial.print(F("/"));
    Serial.print(gps.date.day());
    Serial.print(";;");
    res_flag = true;
  } 

  if (gps.time.isUpdated()) {
    Serial.print(F("Time:"));
    if (gps.time.hour() < 10) Serial.print(F("0"));
    Serial.print(gps.time.hour());
    Serial.print(F(":"));
    if (gps.time.minute() < 10) Serial.print(F("0"));
    Serial.print(gps.time.minute());
    Serial.print(F(":"));
    if (gps.time.second() < 10) Serial.print(F("0"));
    Serial.print(gps.time.second());
    Serial.print(F(":"));
    if (gps.time.centisecond() < 10) Serial.print(F("0"));
    Serial.print(gps.time.centisecond());
    Serial.print(";;");
    res_flag = true;
  } 

  if (res_flag == false){
    Serial.print("n");
  }
}
