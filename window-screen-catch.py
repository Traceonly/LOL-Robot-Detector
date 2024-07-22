import mss
import tkinter as tk
from PIL import Image, ImageTk
import pygetwindow as gw
import time
import numpy as np
import cv2
import pandas as pd
import os
from ultralytics import YOLO

model_path = 'cursorDetector_x.pt'

def load_model(model_path):
    model = YOLO(model_path)
    return model

# 处理单帧图像
def process_frame(frame, model):
    results = model(frame)
    if len(results) > 0 and len(results[0].boxes) > 0:
        highest_conf = 0.5
        for result in results:
            cursor = result.boxes
            xyxy = cursor.xyxy.cpu().numpy()[0]
            conf = cursor.conf.cpu().numpy()[0]
            if conf > highest_conf:
                highest_conf = conf
                x1, y1, x2, y2 = xyxy[:4]
               
        if highest_conf > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 在帧上添加文本显示置信度
            cv2.putText(frame, f'Conf: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            return [x1, y1, x2, y2, highest_conf]
        else:
            return None
        
    return None

#窗口捕捉 
def capture_window_at_fps_mss(window_title, duration, fps=30):
    model=load_model(model_path)
    with mss.mss() as sct:
        # 根据窗口标题获取窗口信息（需要自行实现）
       # monitor = {'top': 160, 'left': 160, 'width': 800, 'height': 640}  # 示例数值
        window = gw.getWindowsWithTitle(window_title)[0]
        if window:
            if window.isMinimized:  # 检查窗口是否最小化
                window.restore()  # 如果窗口最小化了，则恢复窗口
            monitor = {
                "top": window.top,
                "left": window.left,
                "width": window.width,
                "height": window.height
            }
        else:
            print(f"未找到标题为 '{window_title}' 的窗口。")
            return 
        window.activate()
        root = tk.Tk()
        label = tk.Label(root)
        label.pack()

        start_time = time.time()
        end_time = start_time + duration
        frame_duration = 1 / fps

        # 持续循环处理图像直到达到指定的时间
        while time.time() < end_time:  
            frame_start_time = time.time()  # 记录每一帧开始的时间

            # 使用 grab 方法捕捉指定窗口的屏幕
            screenshot = sct.grab(monitor)
            frame = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)  # 将PIL图像转换为cv2图像

            process_frame(frame,model)#调用帧处理函数
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            tk_image = ImageTk.PhotoImage(image=img)
            label.config(image=tk_image)   # 更新标签的图像属性，用于显示新捕获的屏幕图像
            label.image = tk_image  # 保存图像引用，防止被垃圾回收机制回收
            root.update()  # 更新Tkinter根窗口，以显示新图像

            frame_time = time.time() - frame_start_time  # 计算渲染当前帧所需的时间
            wait_time = frame_duration - frame_time  # 计算需要等待的时间，以保持设定的帧率
            if wait_time > 0:
                time.sleep(wait_time)  # 如果需要，则进行等待
            else:
                print("Warning: Frame rate dropped")  # 如果渲染时间超过了帧间隔，发出帧率下降的警告

        root.destroy()  # 循环结束后销毁Tkinter根窗口
# 使用窗口的标题和持续时间调用函数
capture_window_at_fps_mss('小明剑魔直播_英雄联盟直播_斗鱼直播', duration=1000)  # 持续时间为10秒
