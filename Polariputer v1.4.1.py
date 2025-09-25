    # -----------------------------------------------------------------------------
# Polariputer v1.4.1
# Copyright (C) 2024 Physical Chemistry Lab, College of Chemistry and Molecular Engineering, Peking University
# Authors: Xie Huan, Wuyang Haotian, Chen Zhenyu
#
# This software is provided for academic and education purposes only.
# Unauthorized commercial use is prohibited.
# For inquiries, please contact xujinrong@pku.edu.cn.
# -----------------------------------------------------------------------------


# shortage: LDL: Light-Dark-Light; DLD: Dark-Light-Dark


import cv2
import time
import numpy as np
import serial
import threading
from tkinter import *
from tkinter import messagebox, filedialog
import tkinter as tk
import serial.tools.list_ports
from prettytable import PrettyTable
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from PIL import Image, ImageTk
import matplotlib.pyplot as plt




# Parameter Initialization
arduino_port = None
ser = None
window = Tk()
running = True
link_COM = []
images_0 =[]
images_1 =[]
image = 0
change = 0
close = 0
gaga = 0
n = 1
measure = 0
measure_sta = 0
Motor2Polar_constant = 750 # 1 degree = 750 steps

# Window title
window.title("AI for parimeter")

# Window Size
window.geometry("1350x750")

# Five Sections
label1 = tk.Label(window, text="[Device Control]").place(x=45,y=20)
label2 = tk.Label(window, text="[Model Training]").place(x=275,y=20)
label3 = tk.Label(window, text="[Static Measurement]").place(x=45,y=220)
label4 = tk.Label(window, text="[Dynamic Measurement]").place(x=275,y=220)
label5 = tk.Label(window, text="[Camera Control]").place(x=645,y=20)

# Image processing function, returns a flattened list of the cropped circle image
def process(frame):
    # Covert the frame to gray type 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use Canny edge detection
    edges = cv2.Canny(gray, 30, 50, apertureSize=3)
    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 30, param1=40, param2=10, minRadius=0, maxRadius=100)
    # Output the detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        i = circles[0][0]
        # Determine the center and radius of the circle
        x, y, r = i[0], i[1], i[2]
        # Calculate the boundaries for cropping
        left = max(0, x - r)
        right = min(gray.shape[1], x + r)
        top = max(0, y - r)
        bottom = min(gray.shape[0], y + r)
        # Crop the image
        cropped = gray[top:bottom, left:right]
        # Resize the image to 20x20
        resized = cv2.resize(cropped, (20, 20))
        # Flatten the image and convert it to a 1d numpy array
        flatten_image = np.array(resized).flatten()
        return flatten_image


# Initialize frame counter
frame_count = 0

# Function to get Light-Dark-Light video
def mode0():
    # Read video file from the entry
    video_path = address0_entry.get()
    reader = cv2.VideoCapture(video_path)
    # Loop through each frame of the video
    while(reader.isOpened()):
        try:
            # Read a frame
            ret, frame = reader.read()
            # If success, process the frame and save it as an image
            if ret:
                flatten_image = process(frame)
                images_0.append(flatten_image)
            else:
                messagebox.showinfo("success", "Light-Dark-Light loaded")
                print(len(images_0))
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    # Release the VideoCapture object
    reader.release()


# Function to get Dark-Light-Dark video
def mode1():
    # Read video file from the entry
    video_path = address1_entry.get()
    reader = cv2.VideoCapture(video_path)
    # Loop through each frame of the video
    while (reader.isOpened()):
        try:
            # Read a frame
            ret, frame = reader.read()
            # If success, process the frame and save it as an image
            if ret:
                flatten_image = process(frame)
                images_1.append(flatten_image)
            else:
                messagebox.showinfo("Success", "Dark-Light-Dark loaded")
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    # Release the VideoCapture object
    reader.release()

# Model training function
def training():
    # Define the trained model as a global variable
    global clf2
    # Get the basic dataset path from entry and add the basic dataset
    data0_path = addresst_entry.get()+"/dataset0"
    data1_path= addresst_entry.get()+"/dataset1"
    for filename in os.listdir(data0_path):
        img = Image.open(os.path.join(data0_path, filename))
        if img is not None:
            images_0.append(np.array(img).flatten())
    for filename in os.listdir(data1_path):
        img = Image.open(os.path.join(data1_path, filename))
        if img is not None:
            images_1.append(np.array(img).flatten())
    # Create labels for LDL and DLD datasets
    labels_0 = [0] * len(images_0)
    labels_1 = [1] * len(images_1)
    # Merge the images and labels from both datasets
    images = images_0 + images_1
    labels = labels_0 + labels_1
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train2 = [i/255 for i in X_train]
    X_test2 = [i/255 for i in X_test]
    # Create and train the logistic regression model
    clf2 = LogisticRegression(C=0.05)
    clf2.fit(X_train2,y_train)
    # Predict the labels for the test set
    y_pred2 = clf2.predict(X_test2)
    accuracy = str(round(accuracy_score(y_test, y_pred2), 4))
    messagebox.showinfo("Training finished", "Accuracy: "+accuracy)
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred2)
    # Calculate ROC curve and AUC
    y_score = clf2.predict_proba(X_test2)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    # Create a layer
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Create a confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    # Display the confusion matrix
    disp.plot(ax=axes[0], text_kw={'fontsize': 16}, colorbar=False)
    axes[0].set_xlabel('Predicted label', fontsize=16)
    axes[0].set_ylabel('True label', fontsize=16)
    axes[0].tick_params(axis='both', which='major', labelsize=16, direction='in')
    axes[0].set_title('Confusion Matrix', fontsize=18, fontweight='bold',pad=20)
    # Plot the ROC curve
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=16)
    axes[1].set_ylabel('True Positive Rate', fontsize=16)
    axes[1].tick_params(axis='both', which='major', labelsize=14, direction='in')
    axes[1].tick_params(axis='both', which='minor', labelsize=14, direction='in')
    axes[1].legend(loc="lower right",fontsize=16,frameon=False)
    axes[1].set_title('ROC Curve', fontsize=18, fontweight='bold',pad=20)
    plt.tight_layout()
    plt.show()

# Prediction Function
def process_frame(frame):
    # Use the process function to obtain the flattened image and make a prediction
    flatten_image = process(frame)
    prediction = clf2.predict([i / 255 for i in flatten_image])
    return prediction

# Select file
def select_ldl_file():
    filename = filedialog.askopenfilename(title="Select Light-Dark-Light Videl File")
    address0_entry.delete(0, tk.END)
    address0_entry.insert(0, filename)

def select_dld_file():
    filename = filedialog.askopenfilename(title="Select Light-Dark-Light Videl File")
    address0_entry.delete(0, tk.END)
    address0_entry.insert(0, filename)

def select_data_folder():
    foldername = filedialog.askdirectory(title="Select Basic Training Data Folder")
    addresst_entry.delete(0, tk.END)
    addresst_entry.insert(0, foldername)

def select_save_folder():
    foldername = filedialog.askdirectory(title="Select Save Folder")
    save_path_entry.delete(0, tk.END)
    save_path_entry.insert(0, foldername)

# Entry and label for LDL video path
address0_label = tk.Label(window, text="Light-Dark-Light Video Path (with format)").place(x=280,y=40)
address0_entry = tk.Entry(window)
address0_entry.place(x=280,y=60,width = 180, height = 30)
Button(window, text="Load Light-Dark-Light", command = mode0).place(x=480,y=60)
Button(window, text="...", command=select_ldl_file).place(x=460, y=60)
# Entry and label for DLD video path
address1_label = tk.Label(window, text="Dark-Light-Dark Video Path(with format)").place(x=280,y=90)
address1_entry = tk.Entry(window)
address1_entry.place(x=280,y=110,width = 180, height = 30)
Button(window, text = "Load Dark-Light-Dark",command = mode1).place(x=480,y=110)
Button(window, text="...", command=select_dld_file).place(x=460, y=110)
# Entry and label for basic training data path
addresst_label = tk.Label(window, text="Basic Training Data Path").place(x=280,y=140)
addresst_entry = tk.Entry(window)
addresst_entry.place(x=280,y=160,width = 180, height = 30)
Button(window, text="Train Model", command=training).place(x=480,y=160)
Button(window, text="...", command=select_data_folder).place(x=460, y=160)

# Function to find available COM ports
def find_COM():
    global ser, arduino_port
    ports = serial.tools.list_ports.comports()
    for port in ports:
        link_COM.append(port.device)
    print(link_COM)
    btn_link.config(state=tk.NORMAL)

# Function to link the Arduino board (motor)
def link_start():
    global ser, arduino_port
    # Get all available COM ports before connecting
    ports_new = serial.tools.list_ports.comports()
    # Traverse all ports, find the new port after connecting the board
    for port in ports_new:
        if port.device in link_COM:
            link_COM.remove(port.device)
        else:
            link_COM.append(port.device)
    arduino_port = link_COM[0]
    print("Find Arduino port: {}".format(arduino_port))
    if arduino_port is None:
        messagebox.showerror("Error", "Arduino port not found")
        return
    # Baud rate
    baudRate = 9600
    # Connect to the Arduino board
    ser = serial.Serial(arduino_port, baudRate, timeout=0.5)
    messagebox.showinfo("link success", "Connected to" + arduino_port)

# Button to find COM ports
Button(window, text="Find Port", command=find_COM).place(x=50,y=60)
# Button to link the Arduino board (motor)
btn_link = Button(window, text="Connect Motor", command=link_start, state=tk.DISABLED)
btn_link.place(x=125,y=60)

# Forward rotation function
def rotate1():
    global measure_sta, measure
    # Get the rotation value from the entry
    num = entry.get()
    # If measuring optical rotation
    if measure_sta == 1:
        # Record the forward rotation step count
        measure += float(num)*Motor2Polar_constant
        num = int(num)
        if num == 1:
            ser.write(b'a')
        if num == 2:
            ser.write(b'b')
        if num == 3:
            ser.write(b'c')
        if num == 4:
            ser.write(b'd')
        if num == 5:
            ser.write(b'e')
        time.sleep(1)
    # Cotrol the motor    
    num = int(10*float(num))
    # b'8' corresponds to about 0.1 degree per forward step, num is the number of 0.1 degree steps
    demo8 = b'8'
    for i in range(num):
        ser.write(demo8)

# Reverse rotation function
def rotate2():
    global measure_sta, measure
    # Get the rotation value from the entry
    num = entry.get()
    # If measuring optical rotation
    if measure_sta == 1:
        # Record the forward rotation step count
        measure -= float(num)*Motor2Polar_constant
        num = int(num)
        if num == 1:
            ser.write(b'f')
        if num == 2:
            ser.write(b'g')
        if num == 3:
            ser.write(b'h')
        if num == 4:
            ser.write(b'i')
        if num == 5:
            ser.write(b'j')
        time.sleep(1)
    # Cotrol the motor    
    num = int(10*float(num))
    # b'9' corresponds to about 0.1 degree per reverse step, num is the number of 0.1 degree steps
    demo9 = b'9'
    for i in range(num):
        ser.write(demo9)

# Rotation control section
entry_label = tk.Label(window, text="Rotation Angle").place(x=50,y=100)
# Entry for rotation angle input
entry = tk.Entry(window)
entry.place(x=50,y=120,width=50,height=30)
# Forward rotation button
Button(window, text=' + ', command=rotate1).place(x=110,y=120)
# Reverse rotation button
Button(window, text=' - ', command=rotate2).place(x=150,y=120)

# Function to find the camera
def link_camera():
    global image, running, change, close
    # Connect to the camera
    cap = cv2.VideoCapture(image, cv2.CAP_DSHOW)
    if not (cap.isOpened()):
        print("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))
    while running:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        # If "change" == 1, change the "image", which means change device; if close == 1, close the camera and reset close
        if change == 1:
            image = change
            btn_camera.config(state=tk.NORMAL)
            btn_alpha0.config(state=tk.NORMAL)
            change = 0
            break
        if close == 1:
            btn_camera.config(state=tk.NORMAL)
            btn_alpha0.config(state=tk.NORMAL)
            close = 0
            break
        btn_camera.config(state=tk.DISABLED)
        clo_camera.config(state=tk.NORMAL)
        btn_alpha0.config(state=tk.DISABLED)
        # Covert to RGB format, because OpenCV uses BGR by default
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Use PIL to convert the frame to ImageTk image
        images = Image.fromarray(frame)
        img = ImageTk.PhotoImage(images)
        # Update the image on the label
        lbl.configure(image=img)
        lbl.imgtk = img
        # Wait for 10ms to proceed to the next loop
        window.update()
        cv2.waitKey(10)
    # Release the camera resources
    cap.release()
    cv2.destroyAllWindows()

# Function to change the camera
def another_camera():
    global change
    if change != 1:
        change = 1
    change_camera.config(state=tk.DISABLED)
    clo_camera.config(state=tk.DISABLED)

# Function to close the camera
def close_camera():
    global close
    if close != 1:
        close = 1
    clo_camera.config(state=tk.DISABLED)
    btn_camera.config(state=tk.NORMAL)
    btn_alpha0.config(state=tk.NORMAL)

# Button to connect the camera
btn_camera = Button(window, text="Connect", command=link_camera)
btn_camera.place(x=650,y=60)
# Button to change the camera
change_camera = Button(window, text = "Change", command=another_camera)
change_camera.place(x=720,y=60)
# Button to close the camera
clo_camera = Button(window, text = "Disconnect", command = close_camera, state = tk.DISABLED)
clo_camera.place(x=790, y=60)
# Window label to display the camera feed
lbl = tk.Label(window)
lbl.place(x=650,y=120)

# alpha_0 mode function
def alpha_0():
    global running
    # Disable the alpha_0 button
    btn_alpha0.config(state=tk.DISABLED)
    # Set motor parameters
    demo4 = b"4"
    demo5 = b"5"
    demo6 = b"6"
    demo7 = b"7"
    # Send the command to Arduino to perform pre-rotation
    alpha_0 = ser.write(demo4)
    # Set the decision list
    p_infinite = [2, 2, 2]
    # Set the image list
    inf_image = [0, 0, 0]
    # Connect to the camera
    cap = cv2.VideoCapture(image, cv2.CAP_DSHOW)
    if not (cap.isOpened()):
        print("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))

    while running:
        # Capture a frame from the camera
        ret, frame = cap.read()
        # If the frame is successfully read
        if ret:
            # Covert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Use PIL to convert the frame to an ImageTk image
            images = Image.fromarray(frame)
            img = ImageTk.PhotoImage(images)
            # Update the image on the label
            lbl.configure(image=img)
            lbl.imgtk = img
            # Read data sent from Arduino
            read = ser.readline()
            read = int(read.decode("utf-8"))
            # Update the stored image list
            inf_image.append(frame)
            inf_image.pop(0)
            # If data from Arduino is successfully read, which means pre-rotation or tracking is finished, continue with the next steps
            if read == 1:
                # Update the decision list
                p_infinite.append(process_frame(frame))
                p_infinite.pop(0)
                print(p_infinite)
                # If the decision list reaches the end condition, send the number 7 to return a certain angle
                if p_infinite == [1, 0, 0]:
                    ser.write(demo7)
                    time.sleep(1.5)
                    btn_alpha0.config(state=tk.NORMAL)
                    break
                # If not at the end, send the number 5 to continue tracking with the motor
                else:
                    ser.write(demo5)
            # wait 10 ms before the next loop
            window.update()
            cv2.waitKey(10)
    # Release the camera resources
    cap.release()
    cv2.destroyAllWindows()
    # alpha_0 finished
    messagebox.showinfo("alpha_0", "Zero Point Determined")

# alpha_0 mode button
btn_alpha0 = Button(window, text="Find Zero", command=alpha_0)
btn_alpha0.place(x=50,y=250)

# Manual preparation for optical rotation measurement 
def alpha_mea_start():
    global measure, measure_sta, run
    run = True
    measure = 0
    measure_sta = 1
    btn_alpha_measure2.config(state=tk.NORMAL)
    btn_alpha_measure.config(state = tk.DISABLED)
    # Connect to the camera
    cap = cv2.VideoCapture(image, cv2.CAP_DSHOW)
    if not (cap.isOpened()):
        print("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))
    while run:
        # Capture a frame 
        ret, frame = cap.read()
        # if the frame is successfully read
        if ret:
            # Covert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Use PIL to convert the frame to an ImageTk image
            images = Image.fromarray(frame)
            img = ImageTk.PhotoImage(images)
            # Update the image on the label
            lbl.configure(image=img)
            lbl.imgtk = img
            # Wait for 10 ms before the next loop
            window.update()
            cv2.waitKey(10)

# Button for optical rotation measurement
btn_alpha_measure = Button(window, text = "Measure Rotation", command = alpha_mea_start)
btn_alpha_measure.place(x=50,y=300)

# Motor fine-tuning function for static optical rotation measurement
def alpha_mea_motor():
    global measure, measure_sta, run
    global running
    btn_alpha_measure2.config(state=tk.DISABLED)
    # Set parameters
    demo4 = b"4"
    demo5 = b"5"
    demo6 = b"6"
    demo3 = b"3"
    demo0 = b"0"
    run = False
    # Decision list
    p_measure = [2, 2, 2]
    # Image list
    measure_image = [0, 0, 0]
    # Connect to the camera
    cap = cv2.VideoCapture(image, cv2.CAP_DSHOW)
    if not (cap.isOpened()):
        print("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))

    while running:
        # capture a frame from the camera
        ret, frame = cap.read()
        # if success
        if ret:
            # Covert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Use PIL to convert the frame to an ImageTk image
            images = Image.fromarray(frame)
            img = ImageTk.PhotoImage(images)
            # Update the image on the label
            lbl.configure(image=img)
            lbl.imgtk = img
            # Read data sent from Arduino
            read = ser.readline()
            if read != b'':
                read = int(read.decode("utf-8"))
            # Update the stored image list
            measure_image.append(frame)
            measure_image.pop(0)
            # Determine if the first image is LDL(0) or DLD(1)
            if p_measure == [2, 2, 2]:
                prediction_1st = process_frame(frame)
                # Update decision list
                p_measure.append(process_frame(frame))
                p_measure.pop(0)
                # Determine the direction of the first rotation based on the first prediction
                if prediction_1st == 1:
                    ser.write(demo5)
                if prediction_1st == 0:
                    ser.write(demo3)
            # If data from Arduino is successfully read, continue with the next steps
            if read == 1:
                # Update the decision list
                p_measure.append(process_frame(frame))
                p_measure.pop(0)
                # If rotate in reverse and reach the end condition
                if p_measure == [1, 0, 0] and prediction_1st == 1:
                    ser.write(demo6)                                           # If the end condition is reached, send 6 to return a certain angle
                    time.sleep(1.5)
                    # Read alpha_measure data sent from Arduino
                    alpha_measure = ser.readline()
                    # Adjust the format and sign
                    alpha_measure = -float(alpha_measure.decode("utf-8"))
                    alpha_measure = alpha_measure + measure
                    print(ser.readline())
                    # Display the measurement result
                    res_measure.config(text=str(int(alpha_measure)) + " steps")
                    # Initialize
                    measure = 0
                    measure_sta = 0
                    btn_alpha_measure.config(state=tk.NORMAL)
                    break
                # If rotate forward and reach the end condition
                if p_measure == [0, 1, 1] and prediction_1st == 0:
                    ser.write(demo0)
                    time.sleep(1.5)
                    # Read alpha_measure data sent from Arduino
                    alpha_measure = ser.readline()
                    # Adjust the format and sign
                    alpha_measure = float(alpha_measure.decode("utf-8"))
                    alpha_measure = alpha_measure + measure
                    # Display the measurement result
                    res_measure.config(text=str(int(alpha_measure)) + " steps")
                    # Initialize
                    measure = 0
                    measure_sta = 0
                    btn_alpha_measure.config(state=tk.NORMAL)
                    break
                # If not at the end, send the corresponding number for forward/reverse rotation to continue tracking
                if prediction_1st == 1:
                    ser.write(demo5)
                if prediction_1st == 0:
                    ser.write(demo3)
            # Wait 10ms before the next loop
            window.update()
            cv2.waitKey(10)
    cap.release()
    cv2.destroyAllWindows()

# Button to start optical rotation measurement
btn_alpha_measure2 = Button(window, text = "Begin", command = alpha_mea_motor,state=tk.DISABLED)
btn_alpha_measure2.place(x=180, y=300)
# Display result
label_measure = tk.Label(window, text="Result: ").place(x=50,y=350)
res_measure = Label(window, text="")
res_measure.place(x=110,y=350)

# alpha_t Function
def exstart():
    global num_start
    # Lock the Start Experiment buttion
    btn_alpha_t_start.config(state=tk.DISABLED)
    demo3 = b"3"
    demo8 = b"8"  # Forward rotation
    demo9 = b"9"  # Reverse rotation
    # Get the preset angle from the entry
    num_start = start_entry.get()
    num_start = 10*int(num_start)
    # If positive, rotate forward; vice versa
    if num_start > 0:
        for i in range(num_start):
            ser.write(demo8)
    if num_start < 0:
        num_start2 = -num_start
        for i in range(num_start2):
            ser.write(demo9)
    # Show timing start button
    btn_alpha_t_time.config(state=tk.NORMAL)

# Timing start function
def altime():
    # Define global variables
    global start, running, gaga
    # Disable the Timing Start button
    btn_alpha_t_time.config(state=tk.DISABLED)
    # Record the experiment start time
    start = time.time()
    # Connect to the camera
    cap = cv2.VideoCapture(image, cv2.CAP_DSHOW)
    if not (cap.isOpened()):
        print("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))
    # Show the recognition start button 
    btn_alpha_t_ml.config(state=tk.NORMAL)
    # Popup for successful timing start
    messagebox.showinfo("alpha_t", "Timing Start")
    while running:
        # Capture a frame from the camera
        ret, frame = cap.read()
        # If succeed
        if ret:
            # Covert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Use PIL to convert the frame to an ImageTk image
            images = Image.fromarray(frame)
            img = ImageTk.PhotoImage(images)
            # Update the image on the label
            lbl.configure(image=img)
            lbl.imgtk = img
            # If recognition is started, stop running this function
            if gaga == 1:
                gaga = 0
                break
            # Wait for 10 ms before the next loop
            window.update()
            cv2.waitKey(10)
    # Release the camera resources
    cap.release()
    cv2.destroyAllWindows()

# Function to display results
def display_table(table):
    # Clear the data in the Listbox
    listbox.delete(0, tk.END)
    # Covert PrettyTable data to a list of strings
    table_lines = table.get_string().split('\n')
    # Add each line of the table to the listbox
    for line in table_lines:
        listbox.insert(tk.END, line)

# Recognition start function
def ML():
    global running, gaga, n
    # Disable the Recognition Start button
    btn_alpha_t_ml.config(state=tk.DISABLED)
    time.sleep(2)
    # Determine the number of samples to collect
    num_sample = int(sample_entry.get())
    # Result list to store time and step data
    result = []
    # Image storage list
    res_image = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Time storage list
    p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Decision list
    p2 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # Parameter settings
    gaga = 1
    step = int(1 * num_start * Motor2Polar_constant/10)
    demo0 = b"0"
    demo1 = b"1"
    demo2 = b"2"
    demo3 = b"3"
    # Connect to the camera
    cap = cv2.VideoCapture(image, cv2.CAP_DSHOW)
    if not (cap.isOpened()):
        print("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))

    # If experiment starts successfully, show a message box
    messagebox.showinfo("success", "Experiment Start")
    guagua = 1
    while running and guagua:
        # capture a frame from the camera
        ret, frame = cap.read()
        # If the frame is successfully read
        if ret:
            # Covert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Use PIL to convert the frame to an ImageTk image
            images = Image.fromarray(frame)
            img = ImageTk.PhotoImage(images)
            # Update the image on the label
            lbl.configure(image=img)
            lbl.imgtk = img
        # Reaction time for the corresponding frame
        current_time = time.time() - start
        current_time = round(current_time, 2)
        # Bind the variable to display time
        time_var.set(f"{current_time:.2f}")
        # Update the decision list
        p.append(current_time)
        p.pop(0)
        p2.append(process_frame(frame))
        p2.pop(0)
        res_image.append(frame)
        res_image.pop(0)
        # If the decision list meets the requirement
        if p2 == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
            # Save the time of the first element in the list and record the step count
            result.append((p[0], step))
            # Update the step count
            step -= int(0.5*Motor2Polar_constant)
            # Control the motor to rotate half of Motor2Polar constant steps, corresponding to 0.5 degrees
            ser.write(demo1)
            # Update the interface data (actually has no effect)
            listbox.insert(tk.END, result[-1])
            time.sleep(1)
        # If "num_sample" data points are received
        if len(result) == num_sample:
            # Get the path to save the data
            excel_filename = os.path.join(save_path_entry.get(),"output1.xlsx")
            if os.path.exists(excel_filename):
                n = n + 1
                excel_filename = os.path.join(save_path_entry.get(),"output"+str(n)+".xlsx")
            # Reset the motor
            ser.write(demo2)
            # Draw the data table in the box plot
            table = PrettyTable()
            table.add_column("time", [row[0] for row in result])
            table.add_column("steps", [row[1] for row in result])
            display_table(table)
            print(table)
            # Convert the result to a DataFrame
            df = pd.DataFrame(result,columns=['time', 'steps'])
            # Save the DataFrame as an Excel file
            df.to_excel(excel_filename, index=False)
            # Data saved successfully prompt
            messagebox.showinfo("save success", "Data saved to" + excel_filename)
            # Initialize button state
            btn_alpha_t_start.config(state=tk.NORMAL)
            guagua = 0
        # Wait 10ms before the next loop
        window.update()
        cv2.waitKey(10)
    # Release camera resources
    cap.release()
    cv2.destroyAllWindows()
    # Exit function
    return

# Function to stop the experiment
def on_close():
    window.destroy()
    if ser is not None:
        ser.close()  # Close the serial port connection when the program ends

# Multithreading
def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()

# Preset angle label
start_label = Label(window, text="Preset Angle").place(x=280,y=310)
# Preset angle entry
start_entry = Entry(window)
# Default preset angle is 5 degree
start_entry.insert(5,"5")
# Preset angle can be changed
# start_entry.configure(bg="grey")
# start_entry.configure(state='disabled')
start_entry.place(x=360,y=310,height=20,width=25)
# Data volume label
sample_label = Label(window, text="Volume").place(x=400,y=310)
# Data volume entry
sample_entry = Entry(window)
# Default number of data points is 12
sample_entry.insert(index=12,string='12')
sample_entry.place(x=450,y=310,height=20,width=25)
# Reaction time label
time_label = Label(window, text="Time(s)").place(x=485, y =310)
# Reaction time variable
time_var = tk.StringVar()
# Reaction time entry
time_entry = Entry(window,textvariable=time_var)
time_entry.place(x=535,y=310,height = 20,width=40)
# Start experiment button
btn_alpha_t_start = Button(window, text="Exp. Start", command=exstart)
btn_alpha_t_start.place(x=280,y=345)
# Timing start button
btn_alpha_t_time = Button(window, text="Timing Start", command=altime, state=tk.DISABLED)
btn_alpha_t_time.place(x=360,y=345)
# Recognition start button
btn_alpha_t_ml = Button(window, text="Recognition Start", command=lambda: thread_it(ML), state=tk.DISABLED)
btn_alpha_t_ml.place(x=460, y=345)
# Software terminate button
close_button = tk.Button(window, text="Software Terminate", command=on_close)
close_button.place(x=280,y=390)
# Create a listbox to display data
listbox = tk.Listbox(window,height=20,width=80)
listbox.place(x=50,y=440)
# Create a scrollbar
scrollbar = tk.Scrollbar(window, command=listbox.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
listbox.config(yscrollcommand=scrollbar.set)
# Data saving path label and entry
save_path_label = tk.Label(window, text="Data Saving Path").place(x=280,y=245)
save_path_entry = tk.Entry(window)
save_path_entry.place(x=280, y=265, width=280, height=30)
Button(window, text="...", command=select_save_folder).place(x=560, y=265)
# Display author information
author_label = tk.Label(window, text="Author: Huan Xie, Haotian Wuyang, Zhenyu Chen\nCopyright (C) 2024 College of Chemistry and Molecular Engineering, Peking University",justify=tk.LEFT,padx = 10,anchor = "w")
author_label.pack(side=tk.BOTTOM, fill=tk.X)
# Run the program interface
window.mainloop()



