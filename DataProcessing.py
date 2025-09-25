# -----------------------------------------------------------------------------
# DataProcess Program for Polariputer v1.4.1
# Copyright (C) 2024 Physical Chemistry Lab, College of Chemistry and Molecular Engineering, Peking University
# Authors: Xie Huan, Wuyang Haotian, Chen Zhenyu

# This software is provided for academic and education purposes only.
# Unauthorized commercial use is prohibited.
# For inquiries, please contact xujinrong@pku.edu.cn.
# -----------------------------------------------------------------------------


import cv2
import numpy as np

# Define the video file path (modify as needed)
video_path = "dist/VideoName.avi" # Other video format is also supported, such as mp4, mov, etc.
# Read the video file
reader = cv2.VideoCapture(video_path)
# Get the video's frame rate, total frame count, and duration
fps = reader.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")
total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")
duration = total_frames / fps
print(f"Duration: {duration} seconds")


def process_frame(frame, output_filename_prefix, frame_count):
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
        # Save the processed image
        cv2.imwrite('{}{}.png'.format(output_filename_prefix, frame_count), resized)

# Initialize frame count
frame_count = 0

# Loop through the video frames
while(reader.isOpened()):
    try:
        # Read a frame from the video
        ret, frame = reader.read()
        # If the frame is read successfully, process it
        if ret:
            process_frame(frame, 'data_5_', frame_count)
            frame_count += 1
        else:
            break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Release the VideoCapture object
reader.release()


