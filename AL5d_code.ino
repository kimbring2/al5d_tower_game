//#if ARDUINO >= 100
#include "Arduino.h"
//#else
//#include "WProgram.h"
//#end if

#include <Servo.h>
#include <math.h>

//comment to disable the Force Sensitive Resister on the gripper
//#define FSRG

//Select which arm by uncommenting the corresponding line
//#define AL5A
//#define AL5B
#define AL5D

//uncomment for digital servos in the Shoulder and Elbow
//that use a range of 900ms to 2100ms
//#define DIGITAL_RANGE

#ifdef AL5A
const float A = 3.75;
const float B = 4.25;
#elif defined AL5B
const float A = 4.75;
const float B = 5.00;
#elif defined AL5D
const float A = 5.75;
const float B = 7.375;
#endif

//Arm Servo pins
#define Base_pin 2
#define Shoulder_pin 3
#define Elbow_pin 4
#define Wrist_pin 10
#define Gripper_pin 11
#define WristR_pin 12

//Onboard Speaker
#define Speaker_pin 5

//Radians to Degrees constant
const float rtod = 57.295779;

//Arm Speed Variables
float Speed = 1.0;
int sps = 3;

//Servo Objects
Servo Elb;
Servo Shldr;
Servo Wrist;
Servo Base;
Servo WristR;
Servo Gripper;

//boolean mode = true;

//Here's all the Inverse Kinematics to control the arm
int Arm_angle(int base_angle, int shldr_angle, int elb_angle, int wrist_angle, int gripper_angle, int wrist_rotata_angle) 
{
  
#ifdef DIGITAL_RANGE
  Elb.writeMicroseconds(map(180 - Elbow, 0, 180, 900, 2100));
  Shldr.writeMicroseconds(map(Shoulder, 0, 180, 900, 2100));
#else
  Elb.write(elb_angle);
  Shldr.write(shldr_angle);
#endif
  Wrist.write(wrist_angle);
  Base.write(base_angle);
  WristR.write(wrist_rotata_angle);
  
  Gripper.write(180 - gripper_angle);

  return 0; 
}

void setup()
{
  Serial.begin(9600);
  Base.attach(Base_pin);
  Shldr.attach(Shoulder_pin);
  Elb.attach(Elbow_pin);
  Wrist.attach(Wrist_pin);
  Gripper.attach(Gripper_pin);
  WristR.attach(WristR_pin);
  Arm_angle(0, 0, 0, 0, 0, 0);
}

int angleBase = 60;
int angleShldr = 135;
int angleElb = 10;
int angleWrist = 90;
int angleGripper = 120;
int angleWristRotate = 0;
long lastReferenceTime;
unsigned char action;

#define actionBaseUp 119                 // w
#define actionBaseDown 115               // s

#define actionShldrUp 97                 // a
#define actionShldrDown 100              // d

#define actionElbUp 101                  // e
#define actionElbDown 113                // q

#define actionWristUp 114                // r
#define actionWristDown 116              // t

#define actionGripperUp 122              // z
#define actionGripperDown 120            // x

#define actionWristRotCW 103             // g
#define actionWristRotCCW 102            // f

void loop()
{
  Arm_angle(angleBase, angleShldr, angleElb, angleWrist, angleGripper, angleWristRotate);

  lastReferenceTime = millis();
  while(millis() <= (lastReferenceTime + 100)){};

  if(Serial.available() > 0)
  {
    // Read character
    action = Serial.read();
    if(action > 0)
    {
      // Set action
      switch(action)
      {
        case actionBaseUp:
        angleBase += 2;
        break;

        case actionBaseDown:
        angleBase -= 2;
        break;
        
        case actionShldrUp:
        angleShldr += 2;
        break;

        case actionShldrDown:
        angleShldr -= 2;
        break;
        
        case actionElbUp:
        angleElb += 2;
        break;

        case actionElbDown:
        angleElb -= 2;
        break;
        
        case actionWristUp:
        angleWrist += 2;
        break;

        case actionWristDown:
        angleWrist -= 2;
        break;
        
        case actionGripperUp:
        angleGripper += 2;
        break;

        case actionGripperDown:
        angleGripper -= 2;
        break;

        case actionWristRotCW:
        angleWristRotate += 2;
        break;
        
        case actionWristRotCCW:
        angleWristRotate -= 2;
        break;
      }
      
      // Display position
      Serial.print("angleBase = "); Serial.print(angleBase, DEC); Serial.println();  
      Serial.print("angleShldr = "); Serial.print(angleShldr, DEC); Serial.println(); 
      Serial.print("angleElb = "); Serial.print(angleElb, DEC); Serial.println(); 
      Serial.print("angleWrist = "); Serial.print(angleWrist, DEC); Serial.println(); 
      Serial.print("angleGripper = "); Serial.print(angleGripper, DEC); Serial.println();
      Serial.print("angleWristRotate = "); Serial.print(angleWristRotate, DEC); Serial.println();
      Serial.println();
      
      // Move arm
      //Arm(tmpx, tmpy, tmpz, tmpg, tmpwa, tmpwr);
      Arm_angle(angleBase, angleShldr, angleElb, angleWrist, angleGripper, angleWristRotate);
      
      // Pause for 100 ms between actions
      lastReferenceTime = millis();
      while(millis() <= (lastReferenceTime + 100)){};
    }
  }
}
