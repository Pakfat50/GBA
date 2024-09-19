#include <TinyGPSPlus.h>
 
TinyGPSPlus gps;

//#define DEBUG
#define GPS_ENABLE

String delimiter = "\t";

void setup() {
  Serial.begin(115200);
  Serial2.begin(9600,SERIAL_8N1,16,17);
}


void loop()
{
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
  uint32_t sys_time = micros();
  uint32_t sensorValue1 = analogRead(26);
  uint32_t sensorValue2 = analogRead(27);
  // print out the value you read:

  while (Serial2.available() > 0) {
    gps.encode(Serial2.read());
  }
  
  Serial.print(sys_time);
  Serial.print(delimiter);
  Serial.print(sensorValue1);
  Serial.print(delimiter);
  Serial.print(sensorValue2);
  Serial.print(delimiter);

  #ifdef GPS_ENABLE
  displayInfo();
  #else
  delay(2);
  #endif

  
  Serial.println();
  #endif

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
