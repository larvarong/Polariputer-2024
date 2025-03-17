#include <Arduino_BuiltIn.h>

const int dirPin = 7; // Direction pin
const int stepPin = 11; // Step pin
const int STEPS_PER_REV = 373; // Steps per revolution, corresponding to 0.00134*373 about 0.5 degrees
const int STEPS_PER_REV_INF = 6; // Step length for measuring alpha0
int steps = 0;
int steps_inf = 0;
int data;
int back;
int back_inf;
bool runonce = true;
int rotate;
int record = 0;

void setup() {
  // Set as output mode:
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {

data = 0; // Initialize data reading
if (Serial.available() > 0) {
    data = Serial.read();
}
// Read data; 49 corresponds to 1

if (data == 52) { // Read 4 from Python, perform alpha_0 pre-rotation
  digitalWrite(dirPin, LOW); // Rotate forward one degree, 746 steps
  for (int x = 0; x < STEPS_PER_REV * 2; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(100);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(100);
  }
  Serial.println(1); // Send back to Python, indicating pre-rotation is complete
}

if (data == 53) { // Read 5 from Python, perform alpha_0/alpha_inf tracking/measurement
  digitalWrite(dirPin, HIGH); // Rotate backward
  for (int x = 0; x < STEPS_PER_REV_INF; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(100);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(100);
  }
  Serial.println(1); // Send back to Python, indicating tracking is complete
  steps_inf = steps_inf + 1; // Increment tracking count
}

if (data == 51) { // Read 3 from Python, perform measurement
  digitalWrite(dirPin, LOW); // Rotate forward
  for (int x = 0; x < STEPS_PER_REV_INF; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(100);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(100);
  }
  Serial.println(1); // Send back to Python, indicating tracking is complete
  steps_inf = steps_inf + 1; // Increment tracking count
}

if (data == 55) { // Read 7 from Python, reset alpha_0
  digitalWrite(dirPin, LOW);
  for (int x = 0; x < STEPS_PER_REV_INF * 2; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(100);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(100);
  }
  steps_inf = 0; // Clear count to prevent reset disorder
}

if (data == 54) { // Read 6 from Python, perform reset and transmission, alpha_inf data or alpha_measure100 data
  digitalWrite(dirPin, LOW);
  for (int x = 0; x < STEPS_PER_REV_INF * 2; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(100);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(100);
  }
  Serial.println(6 * steps_inf - 2 * 6); // alpha_inf rotation degrees * rotation count - adjustment degrees
  back_inf = 6 * steps_inf - 2 * 6 - 750 * record;
  if (back_inf < 0) {
    back_inf = -back_inf;
    digitalWrite(dirPin, HIGH);
  }
  while (back_inf > 0) {
    int steps_to_move = min(back_inf, 7460);
    for (int x = 0; x < steps_to_move; x++) { // Reset
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(100);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(100);
    }
    back_inf -= steps_to_move;
    delay(100);
  }
  steps_inf = 0;
  record = 0;
}

if (data == 48) { // Read 0 from Python, perform reset and transmission, alpha_measure011 data
  digitalWrite(dirPin, HIGH);
  for (int x = 0; x < STEPS_PER_REV_INF * 2; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(100);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(100);
  }
  Serial.println(6 * steps_inf - 2 * 6); // alpha_inf rotation degrees * rotation count - adjustment degrees
  back_inf = 6 * steps_inf - 2 * 6 + 750 * record;
  while (back_inf > 0) {
    int steps_to_move = min(back_inf, 7460);
    for (int x = 0; x < steps_to_move; x++) { // Reset
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(100);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(100);
    }
    back_inf -= steps_to_move;
    delay(100);
  }
  steps_inf = 0;
  record = 0;
}

if (data == 49) { // Read 1 from Python, measure alpha_t, motor rotates backward 343 steps ~0.5 degrees, steps increase by fixed steps
  digitalWrite(dirPin, HIGH);
  for (int x = 0; x < STEPS_PER_REV; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(100);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(100);
  }
  steps = steps + 1;
}

if (data == 50) { // Read 2 from Python, motor reset
  digitalWrite(dirPin, LOW);
  back = STEPS_PER_REV * (steps - 10);
  for (int x = 0; x < back; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(100);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(100);
  }
  steps = 0;
}

if (data == 57) { // Read 9 from Python, motor rotates backward 75 steps ~0.1 degrees
  digitalWrite(dirPin, HIGH);
  rotate = 75;
  for (int x = 0; x < rotate; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
  }
}

if (data == 56) { // Read 8 from Python, motor rotates forward 75 steps ~0.1 degrees
  digitalWrite(dirPin, LOW);
  rotate = 75;
  for (int x = 0; x < rotate; x++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
  }
}

if (data == 97) {
  record = record + 1;
}

if (data == 98) {
  record = record + 2;
}

if (data == 99) {
  record = record + 3;
}

if (data == 100) {
  record = record + 4;
}

if (data == 101) {
  record = record + 5;
}

if (data == 102) {
  record = record - 1;
}

if (data == 103) {
  record = record - 2;
}

if (data == 104) {
  record = record - 3;
}

if (data == 105) {
  record = record - 4;
}

if (data == 106) {
  record = record - 5;
}
}