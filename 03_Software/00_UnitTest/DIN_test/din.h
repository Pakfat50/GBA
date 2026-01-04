#ifndef DIN_H_
#define DIN_H_


#define DIN_PIN0 2
#define DIN_PIN1 3
#define DIN_PIN2 12
#define DIN_PIN3 10
#define DIN_PIN4 16
#define DIN_PIN5 17
#define DIN_PIN6 5
#define DIN_PIN7 4

void dinInit(void);
uint8_t getChannel(void);
uint8_t getSensorId(void);

#endif