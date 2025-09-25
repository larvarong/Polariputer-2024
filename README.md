# README

## Project Title
Polariputer

## Description
This software provides an integrated GUI for automated optical rotation measurement during sucrose hydrolysis using machine learning. After establishing a physical connection with the polarimeter system, stepper motor system, and camera system, the software can:
1. Control the motor’s rotation to adjust the degree of optical rotation.
2. Train machine learning models using an original dataset and expand the training data by incorporating newly captured videos (.avi).
3. Link to the camera to monitor optical rotation changes in real-time, allowing users to verify model performance.

Once the zero point of the polarimeter is calibrated, the software can measure:

1. Static optical rotation: returning a step number (linearly proportional to optical rotation).
2. Dynamic optical rotation variation: generating a logbook containing each endpoint’s step number and timestamp, with data saved to a specified path.

After each measurement, the motor automatically resets to the zero point, ensuring readiness for subsequent measurements.

## Contents
This package contains:
1. `Polariputer v1.4.1.exe` - The packaged executable created with PyInstaller
2. `Polariputer v1.4.1.py` - The original source code
3. `DataProcessing.py` - Code sample for data processing
4. `Basic_dataset v1.4.1.zip` - The original dataset for model training
5. `alpha_t_English.ino` - Arduino file for step motor in experiment, its format equals to .cpp file
6. `Linear_verification.ino` - Arduino file for step motor in linear verification 
7. `requirement.txt` - Python Dependencies
8. `README.md` - This documentation file

## Prerequisites
Before using the software, ensure you have:
1. Operating System: Windows 10/11 (64-bit). This software does NOT support maxOS or Linux. 
2. Hardware:
   - Minimum Specifications: Intel i7-10750H/equivalent CPU, 8 GB RAM.
   - Configuration: Arduino Uno R3 Microcontroller, Stepper Motor, Motor Driver, Camera, Polarimeter relavent setup.
3. Python [3.12+] & following Python packages (only needed if running from source):
   - numpy>=2.2.6
   - matplotlib>=3.10.3
   - opencv-python>=4.10.0.84
   - pandas>=2.3.1   
   - pillow>=11.3.0
   - prettytable>=3.11.0
   - pyserial>=3.5
   - scikit-learn>=1.7.1

## Installation
### Installation of GUI
#### Using the Packaged Executable
1. Download the entire package
2. Run `Polariputer v1.4.1.exe` directly (no installation required)

#### Running from Source
1. Ensure Python [3.12+] is installed
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main script `Polariputer v1.4.1.py`

### Installation of Arduino code to microcontroller
1. Requirements: Arduino IDE ([Download](https://www.arduino.cc/en/software/)), Arduino-compatible board (Arduino Uno R3), USB cable, `.ino` sketch file.
2. Connection: Plug your Arduino board into computer with USB cable.
3. Select Board & Port: in Arduino IDE, follow Tools - Board - Select you Board (Arduino Uno), then Tools - Port - Select COM port (COM with an arduino label).
4. Open & Upload the code: Make sure the .ino file is under a folder with the same name, open it with Arduino IDE, click Verify button, and then click Upload button.
**Note**: Make sure the linear constant between steps and optical rotation is determined through `Linear_verification.ino` sketch file following **Calibration Protocol** in **Supporting Information**, and change `const int STEPS_PER_REV` in `alpha_t_English.ino` sketch file.

## Usage
Please follow the supporting information of the original article.

## Packing Details
This executable was packed with:
1. PyInstaller version [6.14.2]
2. Packing command used:
  ```
  Pyinstaller -F -i [ico name].ico -w [file name].py
  ```

## Security Note
The executable has been packed with PyInstaller. You can:
1. Run the pre-packaged executable.
2. Or inspect the source code and run it directly after verifying its contents.

## Troubleshooting
Common issues and solutions:
1. Error: "Failed to execute script"
   - Try running from source to see detailed error messages.
   - Ensure all dependencies are installed.
   - Delete **-w** in the packing command when packing the code, then you will get a prompt reporting error when the executable is running.

2. Antivirus false positives
   - Some antivirus programs may flag PyInstaller-packed executables.
   - You can add an exception or run from source instead.

## License
Copyright (C) College of Chemistry and Molecular Engineering, Peking University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Contact
For support or questions, please contact:
Xu, Jinrong
[xujinrong@pku.edu.cn](xujinrong@pku.edu.cn)
