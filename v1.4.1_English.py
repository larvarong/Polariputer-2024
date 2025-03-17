# Author: Xie Huan, Wuyang Haotian, Chen Zhenyu
# Copyright (C) 2024 Physical Chemistry Lab of College of Chemistry and Molecular Engineering
import cv2
import time
import numpy as np
import serial
import threading
from tkinter import *
from tkinter import messagebox
import tkinter as tk
import serial.tools.list_ports
from prettytable import PrettyTable
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from PIL import Image, ImageTk



# 参数初始化
arduino_port = None
ser = None
window = Tk()
running = True
# 界面标题
window.title("AI for parimeter")
# 界面大小
window.geometry("1350x750")
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
# 五个区块标签
label1 = tk.Label(window, text="[Device Control]").place(x=45,y=20)
label2 = tk.Label(window, text="[Model Training]").place(x=275,y=20)
label3 = tk.Label(window, text="[Static Measurement]").place(x=45,y=220)
label4 = tk.Label(window, text="[Dynamic Measurement]").place(x=275,y=220)
label5 = tk.Label(window, text="[Camera Control]").place(x=645,y=20)
# 图像处理函数 输入每一帧图像 返回截取圆后的一维列表
def process(frame):

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 30, 50, apertureSize=3)

    # 使用霍夫圆变换
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 30, param1=40, param2=10, minRadius=0, maxRadius=100)

    # 输出检测到的圆
    if circles is not None:
        circles = np.uint16(np.around(circles))
        i = circles[0][0]
        # 确定圆的中心和半径
        x, y, r = i[0], i[1], i[2]

        # 计算截取的边界
        left = max(0, x - r)
        right = min(gray.shape[1], x + r)
        top = max(0, y - r)
        bottom = min(gray.shape[0], y + r)


        # 截取图像
        cropped = gray[top:bottom, left:right]
        # 调整图像大小为20x20
        resized = cv2.resize(cropped, (20, 20))
        # 平坦化图像，得到一维列表
        flatten_image = np.array(resized).flatten()

        return flatten_image


# 初始化帧计数器
frame_count = 0

# 获取亮暗亮视频函数
def mode0():
    # 从entry中读取avi视频文件路径
    video_path = address0_entry.get()
    reader = cv2.VideoCapture(video_path)
    # 循环读取视频的每一帧
    while(reader.isOpened()):
        try:
            # 读取一帧
            ret, frame = reader.read()

            # 如果读取成功，处理这一帧并保存为图像
            if ret:
                flatten_image = process(frame)
                images_0.append(flatten_image)

            else:
                messagebox.showinfo("success", "Light-Dark_Light loaded")
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    # 释放VideoCapture对象
    reader.release()


# 获取暗亮暗视频函数
def mode1():
    # 读取avi视频文件
    video_path = address1_entry.get()
    reader = cv2.VideoCapture(video_path)
    # 循环读取视频的每一帧
    while (reader.isOpened()):
        try:
            # 读取一帧
            ret, frame = reader.read()

            # 如果读取成功，处理这一帧并保存为图像
            if ret:
                flatten_image = process(frame)
                images_1.append(flatten_image)

            else:
                messagebox.showinfo("Success", "Dark-Light_Dark loaded")
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    # 释放VideoCapture对象
    reader.release()

# 训练模型函数
def training():
    # 将训练模型定义为全局变量
    global clf2
    # 从entry中获取常驻数据集路径，添加常驻数据集
    data0_path = addresst_entry.get()+"\dataset0"
    data1_path= addresst_entry.get()+"\dataset1"
    for filename in os.listdir(data0_path):
        img = Image.open(os.path.join(data0_path, filename))
        if img is not None:
            images_0.append(np.array(img).flatten())
    for filename in os.listdir(data1_path):
        img = Image.open(os.path.join(data1_path, filename))
        if img is not None:
            images_1.append(np.array(img).flatten())
    # 创建标签
    labels_0 = [0] * len(images_0)
    labels_1 = [1] * len(images_1)

    # 合并数据
    images = images_0 + images_1
    labels = labels_0 + labels_1

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train2 = [i/255 for i in X_train]
    X_test2 = [i/255 for i in X_test]
    # 创建并训练模型
    clf2 = LogisticRegression(C=0.05)
    clf2.fit(X_train2,y_train)

    # 预测并评估模型
    y_pred2 = clf2.predict(X_test2)
    accuracy = str(round(accuracy_score(y_test, y_pred2), 4))
    messagebox.showinfo("Training finished", "Accuracy: "+accuracy)



# 按钮的名称、函数和位置
Button(window, text="Load Light-Dark-Light", command = mode0).place(x=480,y=60)
Button(window, text = "Load Dark-Light-Dark",command = mode1).place(x=480,y=110)
Button(window, text="Train Model", command=training).place(x=480,y=160)

# 输入明暗明视频路径的窗口和标签
address0_label = tk.Label(window, text="Light-Dark-Light Video Path(.avi)").place(x=280,y=40)
address0_entry = tk.Entry(window)
address0_entry.place(x=280,y=60,width = 180, height = 30)
# 输入暗明暗视频路径的窗口和标签
address1_label = tk.Label(window, text="Dark-Light-Dark Video Path(.avi)").place(x=280,y=90)
address1_entry = tk.Entry(window)
address1_entry.place(x=280,y=110,width = 180, height = 30)
# 输入常驻数据集路径的窗口和标签
addresst_label = tk.Label(window, text="Basic Training Data Path").place(x=280,y=140)
addresst_entry = tk.Entry(window)
addresst_entry.place(x=280,y=160,width = 180, height = 30)

# 寻找串口函数
def find_COM():
    global ser, arduino_port
    ports = serial.tools.list_ports.comports()
    for port in ports:
        link_COM.append(port.device)
    print(link_COM)
    btn_link.config(state=tk.NORMAL)



#连接电机函数
def link_start():
    global ser, arduino_port
    # 获取所有可用的串口
    ports_new = serial.tools.list_ports.comports()

    # 遍历所有串口，查找连接开发板前后新增的串口
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
    # 波特率
    baudRate = 9600
    # 连接新增的串口
    ser = serial.Serial(arduino_port, baudRate, timeout=0.5)
    messagebox.showinfo("link success", "Connected to" + arduino_port)

# 寻找串口按钮
Button(window, text="Find Port", command=find_COM).place(x=50,y=60)
# 连接电机按钮
btn_link = Button(window, text="Connect Motor", command=link_start, state=tk.DISABLED)
btn_link.place(x=125,y=60)

# 正转函数
def rotate1():
    global measure_sta, measure
    # 获取窗口输入的转动数值
    num = entry.get()

    # 如果测量旋光度,记录正转步进数
    if measure_sta == 1:
        measure += float(num)*750
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
    num = 10*float(num)
    num = int(num)

    # b'8'对应电机每次正转约0.1度，num为电机转动0.1度的次数
    demo8 = b'8'
    for i in range(num):
        ser.write(demo8)

# 负转函数
def rotate2():
    global measure_sta, measure
    # 获取窗口输入的转动数值
    num = entry.get()

    # 如果测量旋光度，记录负转步进数
    if measure_sta == 1:
        measure -= float(num)*750
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
    num = 10*float(num)
    num = int(num)
    # b'9'对应电机每次负转约0.1度，num为电机转动0.1度的次数
    demo9 = b'9'
    for i in range(num):
        ser.write(demo9)

# 转动角度标签
entry_label = tk.Label(window, text="Rotation Angle").place(x=50,y=100)
# 转动角度窗口
entry = tk.Entry(window)
entry.place(x=50,y=120,width=50,height=30)
# 正转按钮
Button(window, text=' + ', command=rotate1).place(x=110,y=120)
# 负转按钮
Button(window, text=' - ', command=rotate2).place(x=150,y=120)


# 连接相机函数
def link_camera():
    global image, running, change, close

    # 连接相机
    cap = cv2.VideoCapture(image, cv2.CAP_DSHOW)
    if not (cap.isOpened()):
        print("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))

    while running:
        # 抓取一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 如果change变量为1则更改image，即更改设备；如果close为1，则关闭相机并初始化close
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
        # 转换成RGB格式，因为OpenCV默认是BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用PIL将帧转换成ImageTk图像
        images = Image.fromarray(frame)
        img = ImageTk.PhotoImage(images)

        # 更新label上的图像
        lbl.configure(image=img)
        lbl.imgtk = img


        # 等待10ms，进行下一次循环
        window.update()
        cv2.waitKey(10)
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# 更改相机函数
def another_camera():
    global change
    if change != 1:
        change = 1
    change_camera.config(state=tk.DISABLED)
    clo_camera.config(state=tk.DISABLED)
# 关闭相机函数
def close_camera():
    global close
    if close != 1:
        close = 1
    clo_camera.config(state=tk.DISABLED)
    btn_camera.config(state=tk.NORMAL)
    btn_alpha0.config(state=tk.NORMAL)
# 连接相机按钮
btn_camera = Button(window, text="Connect", command=link_camera)
btn_camera.place(x=650,y=60)
# 更改相机按钮
change_camera = Button(window, text = "Change", command=another_camera)
change_camera.place(x=720,y=60)
# 关闭相机按钮
clo_camera = Button(window, text = "Disconnect", command = close_camera, state = tk.DISABLED)
clo_camera.place(x=790, y=60)
# 显示画面窗口
lbl = tk.Label(window)
lbl.place(x=650,y=120)

# alpha_0模式函数
def alpha_0():
    global running
    # 令alpha0按钮不可点击
    btn_alpha0.config(state=tk.DISABLED)
    # 传输电机参数
    demo4 = b"4"
    demo5 = b"5"
    demo6 = b"6"
    demo7 = b"7"
    # 传输数字4，进行预转动
    alpha_0 = ser.write(demo4)
    # 设置判断列表
    p_infinite = [2, 2, 2]
    # 设置图片列表
    inf_image = [0, 0, 0]

    # 连接相机
    cap = cv2.VideoCapture(image, cv2.CAP_DSHOW)
    if not (cap.isOpened()):
        print("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))


    # 定义判断函数
    def process_frame(frame):
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 30, 50, apertureSize=3)

        # 使用霍夫圆变换
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 30, param1=40, param2=10, minRadius=0, maxRadius=100)

        # 输出检测到的圆
        if circles is not None:
            circles = np.uint16(np.around(circles))
            i = circles[0][0]
            # 确定圆的中心和半径
            x, y, r = i[0], i[1], i[2]

            # 计算截取的边界
            left = max(0, x - r)
            right = min(gray.shape[1], x + r)
            top = max(0, y - r)
            bottom = min(gray.shape[0], y + r)

            # 截取图像
            cropped = gray[top:bottom, left:right]
            # 调整图像大小为20x20
            resized = cv2.resize(cropped, (20, 20))
            # 将图像展平并使用模型进行预测

            prediction = clf2.predict([i / 255 for i in [resized.flatten()]])
            return prediction

    while running:
        # 抓取一帧
        ret, frame = cap.read()
        # 如果成功读到图像
        if ret:
            # 转换成RGB格式，因为OpenCV默认是BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 使用PIL将帧转换成ImageTk图像
            images = Image.fromarray(frame)
            img = ImageTk.PhotoImage(images)

            # 更新label上的图像
            lbl.configure(image=img)
            lbl.imgtk = img
            # 读取Arduino传入的数据
            read = ser.readline()
            read = int(read.decode("utf-8"))
            # 更新存储的图片列表
            inf_image.append(frame)
            inf_image.pop(0)
            # 如果成功读到Arduino传来的数据，意味着预转动或追踪结束，继续进行后续程序
            if read == 1:
                # 更新判断列表
                p_infinite.append(process_frame(frame))
                p_infinite.pop(0)
                print(p_infinite)
                # 如果达到判断的终点，写入数字7 往回复位一定角度
                if p_infinite == [1, 0, 0]:
                    ser.write(demo7)
                    time.sleep(1.5)
                    btn_alpha0.config(state=tk.NORMAL)
                    break
                # 如果没有达到终点，写入数字5，控制电机继续追踪
                else:
                    ser.write(demo5)
            # 等待10ms，进行下一次循环
            window.update()
            cv2.waitKey(10)
    # 释放相机资源
    cap.release()
    cv2.destroyAllWindows()
    # alpha_0结束
    messagebox.showinfo("alpha_0", "Zero Point Determined")


# alpha_0模式按钮
btn_alpha0 = Button(window, text="Find Zero", command=alpha_0)
btn_alpha0.place(x=50,y=250)

# 测量旋光度函数之手操准备
def alpha_mea_start():
    global measure, measure_sta, run
    run = True
    measure = 0
    measure_sta = 1
    btn_alpha_measure2.config(state=tk.NORMAL)
    btn_alpha_measure.config(state = tk.DISABLED)
    # 连接相机
    cap = cv2.VideoCapture(image, cv2.CAP_DSHOW)
    if not (cap.isOpened()):
        print("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))
    while run:
        # 抓取一帧
        ret, frame = cap.read()
        # 如果成功读到图像
        if ret:
            # 转换成RGB格式，因为OpenCV默认是BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 使用PIL将帧转换成ImageTk图像
            images = Image.fromarray(frame)
            img = ImageTk.PhotoImage(images)

            # 更新label上的图像
            lbl.configure(image=img)
            lbl.imgtk = img
            # 等待10ms，进行下一次循环
            window.update()
            cv2.waitKey(10)

# 测量旋光度按钮
btn_alpha_measure = Button(window, text = "Measure Rotation", command = alpha_mea_start)
btn_alpha_measure.place(x=50,y=300)

# 测量旋光度函数之电机微调
def alpha_mea_motor():
    global measure, measure_sta, run
    global running
    btn_alpha_measure2.config(state=tk.DISABLED)
    # 设置参数
    demo4 = b"4"
    demo5 = b"5"
    demo6 = b"6"
    demo3 = b"3"
    demo0 = b"0"
    run = False
    # 判断列表
    p_measure = [2, 2, 2]
    # 图像列表
    measure_image = [0, 0, 0]

    # 连接相机
    cap = cv2.VideoCapture(image, cv2.CAP_DSHOW)
    if not (cap.isOpened()):
        print("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))


    # 定义判断函数
    def process_frame(frame):
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 30, 50, apertureSize=3)

        # 使用霍夫圆变换
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 30, param1=40, param2=10, minRadius=0, maxRadius=100)

        # 输出检测到的圆
        if circles is not None:
            circles = np.uint16(np.around(circles))
            i = circles[0][0]
            # 确定圆的中心和半径
            x, y, r = i[0], i[1], i[2]

            # 计算截取的边界
            left = max(0, x - r)
            right = min(gray.shape[1], x + r)
            top = max(0, y - r)
            bottom = min(gray.shape[0], y + r)

            # 截取图像
            cropped = gray[top:bottom, left:right]
            # 调整图像大小为20x20
            resized = cv2.resize(cropped, (20, 20))
            # 将图像展平并使用模型进行预测
            prediction = clf2.predict([i / 255 for i in [resized.flatten()]])
            return prediction

    while running:
        # 抓取一帧
        ret, frame = cap.read()
        # 如果成功读到图像
        if ret:
            # 转换成RGB格式，因为OpenCV默认是BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 使用PIL将帧转换成ImageTk图像
            images = Image.fromarray(frame)
            img = ImageTk.PhotoImage(images)

            # 更新label上的图像
            lbl.configure(image=img)
            lbl.imgtk = img
            # 读取Arduino传输的数字
            read = ser.readline()
            if read != b'':
                read = int(read.decode("utf-8"))
            # 更新图像列表
            measure_image.append(frame)
            measure_image.pop(0)
            # 判断首个图像为0还是1
            if p_measure == [2, 2, 2]:
                prediction_1st = process_frame(frame)
                # 更新判断列表
                p_measure.append(process_frame(frame))
                p_measure.pop(0)
                # 根据首个判断结果确定第一次转动的方向
                if prediction_1st == 1:
                    ser.write(demo5)
                if prediction_1st == 0:
                    ser.write(demo3)
            # 如果成功读到Arduino传来的数据，意味着分步追踪结束，继续进行后续程序
            if read == 1:
                # 更新判断列表
                p_measure.append(process_frame(frame))
                p_measure.pop(0)
                # 如果达到判断的终点，写入数字6 往回复位一定角度
                # 如果一直在负转
                if p_measure == [1, 0, 0] and prediction_1st == 1:
                    ser.write(demo6)
                    time.sleep(1.5)
                    # 读取Arduino传入的alpha_measure数据
                    alpha_measure = ser.readline()
                    # 调整格式和正负号
                    alpha_measure = -float(alpha_measure.decode("utf-8"))
                    alpha_measure = alpha_measure + measure
                    print(ser.readline())
                    # 显示测量结果
                    res_measure.config(text=str(int(alpha_measure)) + " steps")
                    # 初始化
                    measure = 0
                    measure_sta = 0
                    btn_alpha_measure.config(state=tk.NORMAL)
                    break
                # 如果一直在正转
                if p_measure == [0, 1, 1] and prediction_1st == 0:
                    ser.write(demo0)
                    time.sleep(1.5)
                    # 读取Arduino传入的alpha_measure数据
                    alpha_measure = ser.readline()
                    # 调整格式和正负号
                    alpha_measure = float(alpha_measure.decode("utf-8"))
                    alpha_measure = alpha_measure + measure
                    # 显示测量结果
                    res_measure.config(text=str(int(alpha_measure)) + " steps")
                    # 初始化
                    measure = 0
                    measure_sta = 0
                    btn_alpha_measure.config(state=tk.NORMAL)
                    break
                # 如果没有达到终点，写入正负转对应数字，控制电机继续追踪
                if prediction_1st == 1:
                    ser.write(demo5)
                if prediction_1st == 0:
                    ser.write(demo3)
            # 等待10ms，进行下一次循环
            window.update()
            cv2.waitKey(10)
    cap.release()
    cv2.destroyAllWindows()

# 开始测量按钮
btn_alpha_measure2 = Button(window, text = "Begin", command = alpha_mea_motor,state=tk.DISABLED)
btn_alpha_measure2.place(x=180, y=300)
# 显示结果
label_measure = tk.Label(window, text="Result: ").place(x=50,y=350)
res_measure = Label(window, text="")
res_measure.place(x=110,y=350)

# alpha_t函数
def exstart():
    global num_start
    # 关闭开始实验按钮
    btn_alpha_t_start.config(state=tk.DISABLED)
    demo3 = b"3"
    demo8 = b"8"  # 正转
    demo9 = b"9"  # 负转
    # 获取窗口输入的预转动角度
    num_start = start_entry.get()
    num_start = 10*int(num_start)
    # 若为正值即正转，vice versa
    if num_start > 0:
        for i in range(num_start):
            ser.write(demo8)
    if num_start < 0:
        num_start2 = -num_start
        for i in range(num_start2):
            ser.write(demo9)
    # 出现开始计时按钮
    btn_alpha_t_time.config(state=tk.NORMAL)
# 开始计时按钮
def altime():
    # 初始参数
    global start, running,gaga
    # 关闭开始计时按钮
    btn_alpha_t_time.config(state=tk.DISABLED)
    # 记录实验开始时间
    start = time.time()
    # 连接相机
    cap = cv2.VideoCapture(image, cv2.CAP_DSHOW)
    if not (cap.isOpened()):
        print("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))
    # 出现开始识别按钮
    btn_alpha_t_ml.config(state=tk.NORMAL)
    # 成功开始计时弹窗
    messagebox.showinfo("alpha_t", "Timing Start")
    while running:
        # 抓取一帧
        ret, frame = cap.read()
        # 如果成功独到图像
        if ret:

            # 转换成RGB格式，因为OpenCV默认是BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 使用PIL将帧转换成ImageTk图像
            images = Image.fromarray(frame)
            img = ImageTk.PhotoImage(images)

            # 更新label上的图像
            lbl.configure(image=img)
            lbl.imgtk = img
            # 如果开始识别，停止运行该函数
            if gaga == 1:
                gaga = 0
                break
            # 等待10ms，进行下一次循环
            window.update()
            cv2.waitKey(10)

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()

# 展示结果函数
def display_table(table):
    # 清空Listbox中的数据
    listbox.delete(0, tk.END)

    # 将PrettyTable的数据转换为字符串列表
    table_lines = table.get_string().split('\n')

    # 将每一行数据添加到Listbox中
    for line in table_lines:
        listbox.insert(tk.END, line)


# 开始识别函数
def ML():
    global running,gaga,n
    # 关闭开始识别按钮
    btn_alpha_t_ml.config(state=tk.DISABLED)
    gaga = 1
    time.sleep(2)
    num_sample = int(sample_entry.get())
    # 开始更新时间
    result = []
    # 图像存储列表
    res_image = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 时间记录列表
    p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 判断列表
    p2 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # 参数设置
    demo0 = b"0"
    demo1 = b"1"
    step = int(1 * num_start * 750/10)
    demo2 = b"2"
    demo3 = b"3"
    # 连接相机
    cap = cv2.VideoCapture(image, cv2.CAP_DSHOW)
    if not (cap.isOpened()):
        print("Could not open video device")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))

    # 定义判断函数
    def process_frame(frame):
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 30, 50, apertureSize=3)

        # 使用霍夫圆变换
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 30, param1=40, param2=10, minRadius=0, maxRadius=100)

        # 输出检测到的圆
        if circles is not None:
            circles = np.uint16(np.around(circles))
            i = circles[0][0]
            # 确定圆的中心和半径
            x, y, r = i[0], i[1], i[2]

            # 计算截取的边界
            left = max(0, x - r)
            right = min(gray.shape[1], x + r)
            top = max(0, y - r)
            bottom = min(gray.shape[0], y + r)

            # 截取图像
            cropped = gray[top:bottom, left:right]
            # 调整图像大小为20x20
            resized = cv2.resize(cropped, (20, 20))
            # 将图像展平并使用模型进行预测

            prediction = clf2.predict([i / 255 for i in [resized.flatten()]])
            return prediction
    # 成功开始实验
    messagebox.showinfo("success", "Experiment Start")
    guagua = 1
    while running and guagua:
        # 抓取一帧
        ret, frame = cap.read()
        # 如果成功读到图像
        if ret:
            # 转换成RGB格式，因为OpenCV默认是BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 使用PIL将帧转换成ImageTk图像
            images = Image.fromarray(frame)
            img = ImageTk.PhotoImage(images)

            # 更新label上的图像
            lbl.configure(image=img)
            lbl.imgtk = img
        # 对应帧的反应时间
        current_time = time.time() - start
        current_time = round(current_time, 2)
        # 绑定显示时间的变量
        time_var.set(f"{current_time:.2f}")
        # 更新判断列表
        p.append(current_time)
        p.pop(0)
        p2.append(process_frame(frame))
        p2.pop(0)
        res_image.append(frame)
        res_image.pop(0)
        # 如果判断列表达到要求
        if p2 == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
            # 保存列表中第一个元素的时间并记录步进数
            result.append((p[0], step))
            # 更新步进数
            step -= 373
            # 控制电机转动373步，对应0.5度
            ser.write(demo1)
            # 更新界面数据（实际无法起到作用）
            listbox.insert(tk.END, result[-1])
            time.sleep(1)
        #如果数据点收到num_sample个
        if len(result) == num_sample:
            # 获取保存数据的路径
            excel_filename = os.path.join(save_path_entry.get(),"output1.xlsx")
            if os.path.exists(excel_filename):
                n = n + 1
                excel_filename = os.path.join(save_path_entry.get(),"output"+str(n)+".xlsx")
            # 令电机复位
            ser.write(demo2)
            # 绘制框图中的数据表格
            table = PrettyTable()
            table.add_column("time", [row[0] for row in result])
            table.add_column("steps", [row[1] for row in result])
            display_table(table)
            print(table)
            # 将结果转换为DataFrame
            df = pd.DataFrame(result,columns=['time', 'steps'])
            # 将DataFrame保存为Excel文件
            df.to_excel(excel_filename, index=False)
            # 数据保持成功提示
            messagebox.showinfo("save success", "Data saved to" + excel_filename)
            # 初始化按钮状态
            btn_alpha_t_start.config(state=tk.NORMAL)
            guagua = 0
        # 等待10ms，进行下一次循环
        window.update()
        cv2.waitKey(10)
    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()
    # 退出函数
    return

# 停止实验函数
def on_close():
    window.destroy()
    ser.close()  # 在程序结束时关闭串口连接

# 多线程
def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()

# 预转动标签
start_label = Label(window, text="Preset Angle").place(x=280,y=310)
# 预转动窗口
start_entry = Entry(window)
# 默认预转动5度
start_entry.insert(5,"5")
# 设置预转动角度可以更改
# start_entry.configure(bg="grey")
# start_entry.configure(state='disabled')
start_entry.place(x=360,y=310,height=20,width=25)
# 数据数目标签
sample_label = Label(window, text="Volume").place(x=400,y=310)
# 数据数目窗口
sample_entry = Entry(window)
# 默认数据点数12个
sample_entry.insert(index=12,string='12')
sample_entry.place(x=450,y=310,height=20,width=25)
# 反应时间标签
time_label = Label(window, text="Time(s)").place(x=485, y =310)
# 反应时间变量
time_var = tk.StringVar()
# 反应时间窗口
time_entry = Entry(window,textvariable=time_var)
time_entry.place(x=535,y=310,height = 20,width=40)
# 开始实验按钮
btn_alpha_t_start = Button(window, text="Exp. Start", command=exstart)
btn_alpha_t_start.place(x=280,y=345)
# 开始计时按钮
btn_alpha_t_time = Button(window, text="Timing Start", command=altime, state=tk.DISABLED)
btn_alpha_t_time.place(x=360,y=345)
# 开始识别按钮
btn_alpha_t_ml = Button(window, text="Recognition Start", command=lambda: thread_it(ML), state=tk.DISABLED)
btn_alpha_t_ml.place(x=460, y=345)
# 停止实验按钮
close_button = tk.Button(window, text="Software Terminate", command=on_close)
close_button.place(x=280,y=390)
# 创建一个Listbox来显示数据
listbox = tk.Listbox(window,height=20,width=80)
listbox.place(x=50,y=440)

# 创建滚动条
scrollbar = tk.Scrollbar(window, command=listbox.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
listbox.config(yscrollcommand=scrollbar.set)
#存储路径标签及窗口
save_path_label = tk.Label(window, text="Data Saving Path").place(x=280,y=245)
save_path_entry = tk.Entry(window)
save_path_entry.place(x=280, y=265, width=280, height=30)
#显示作者信息
author_label = tk.Label(window, text="Author: Huan Xie, Haotian Wuyang, Zhenyu Chen\nCopyright (C) 2024 College of Chemistry and Molecular Engineering, Peking University",justify=tk.LEFT,padx = 10,anchor = "w")
author_label.pack(side=tk.BOTTOM, fill=tk.X)
# 程序界面运行
window.mainloop()



