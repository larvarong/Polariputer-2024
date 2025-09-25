// -----------------------------------------------------------------------------
// Linear Verification Program for Polariputer v1.4.1
// Copyright (C) 2024 Physical Chemistry Lab, College of Chemistry and Molecular Engineering, Peking University
// Authors: Xie Huan, Wuyang Haotian, Chen Zhenyu

// This software is provided for academic and education purposes only.
// Unauthorized commercial use is prohibited.
// For inquiries, please contact xujinrong@pku.edu.cn.
// -----------------------------------------------------------------------------


#include <Arduino_BuiltIn.h>

const int dirPin = 7; // Direction pin
const int stepPin = 11; // Step pin
int steps = 375; // Define steps. Number of steps for one operation (adjust as needed)
int stepDelay = 100; // Define step speed. Delay between steps in microseconds (adjust as needed)
bool runonce = false; // Flag to ensure the setup runs only once

void setup(){
    // Set as output mode:
    pinMode(stepPin, OUTPUT);
    pinMode(dirPin, OUTPUT);

    // Define rotate direction 
    digitalWrite(dirPin, LOW);  // Default direction is LOW
    // digitalWrite(dirPin, LOW); // Uncomment this line if you want to set the default direction to HIGH
}

void loop(){
  if (!runonce) {
    for (int x = 0; x < steps; x++) {
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(stepDelay);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(stepDelay);
    }
    runonce = true; // Flag to prevent re-running the setup
  }
}