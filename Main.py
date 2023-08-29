#国赛2.0版本

#支持库
from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile
from PySide2.QtGui import *
import PySide2

import threading
# from PIL import Image
import cv2 as cv
import numpy as np

import time

import pyrealsense2 as rs
from detect_one import yolo_detect

from models.experimental import attempt_load
from utils.torch_utils import select_device, load_classifier, time_synchronized

from tracker import update_tracker

import pandas as pd
from collections import Counter

txt_number = 0

#运行界面 显示图像
class Stats:

    def __init__(self):
        # 从文件中加载UI定义
        qfile_stats = QFile("ui/mainwindow.ui")
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()

        self.ui = QUiLoader().load('ui/mainwindow.ui')

        self.device = select_device('0')
        self.model = attempt_load('weights/3D_detect.pt')

        self.ui.button.clicked.connect(self.Start)


    def Start(self):

        # self.cap = cv.VideoCapture(0)
        # self.frameRate = self.cap.get(cv.CAP_PROP_FPS)
        ##>>>>>>>>>>>>>>>>>>>>>此处加入深度摄像头初始化，单线读取 两个图像分别连接至不同的Qlable槽<<<<<<<<<<<<<<<<<<<<

        # 创建视频显示线程

        th = threading.Thread(target=self.read_img)
        th.start()

        # th = threading.Thread(target=self.Display_2)
        # th.start()

        self.ui.textBrowser_1.append('开始运行。')


    # 单线主窗，小窗同时显示 并计算帧率
    def read_img(self):

        global txt_number

        txt_number = txt_number + 1

        # 创建摄像头显示线程
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = pipeline.start(config)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Color Intrinsics
        intr = color_frame.profile.as_video_stream_profile().intrinsics

        align_to = rs.stream.color
        align = rs.align(align_to)

        nowtime_1 = time.time()  # 起始时间
        img_count = 0

        # >>>>>>>>>>>>>创建列表保存结果<<<<<<<<<<<<
        all_target = []
        all_id = []

        last_time_output = time.time()

        while True:
            now_time_output = time.time()

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()  # 读取深度数据

            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                print('丢失帧')
                continue

            # part2 预处理

            # 渲染深度图
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

            # convertScaleAbs 将图像归一化转换到int8类型
            # applyColorMap 伪色彩函数 灰度映射到对应的颜色

            depth_img_resize = cv.resize(depth_colormap, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
            depth_img_BGR = cv.cvtColor(depth_img_resize, cv.COLOR_RGB2BGR)

            color_image = np.asanyarray(color_frame.get_data())
            color_image_BGR = cv.cvtColor(color_image, cv.COLOR_RGB2BGR)

            # print('The type of frames is :' , type(color_image))
            # qimage_color = QImage(color_image , 640 , 480 , QImage.Format_BGR888)
            # pixmap_color = QPixmap.fromImage(qimage_color)
            # self.dis_update.emit(pixmap)

            # deep_image = (255 << 24 | depth_img_resize[:, :, 0] << 16 |  depth_img_resize[:, :, 1] << 8 |  depth_img_resize[:, :,2])  # pack RGB values
            deep_img_QImage = PySide2.QtGui.QImage(depth_img_BGR, 320, 240, PySide2.QtGui.QImage.Format_RGB888)
            # deep_img_QImage = PySide2.QtGui.QImage(deep_image, 320, 240, PySide2.QtGui.QImage.Format_RGB32)
            self.ui.label_2.setPixmap(QPixmap.fromImage(deep_img_QImage))
            self.ui.label_2.setScaledContents(True)

            # color_img = (255 << 24 | color_image[:, :, 0] << 16 | color_image[:, :, 1] << 8 | color_image[:, :,2])  # pack RGB values
            # color_img_QImage = PySide2.QtGui.QImage(color_image_BGR, 640, 480, PySide2.QtGui.QImage.Format_RGB888)
            # self.ui.label_1.setPixmap(QPixmap.fromImage(color_img_QImage))
            # self.ui.label_1.setScaledContents(True)

            img_count = img_count + 1  # 计数器

            nowtime_2 = time.time()  # 截止时间

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>帧率计算<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            if (nowtime_2 - nowtime_1 >= 1.2):
                nowtime_1 = nowtime_2
                FPS = img_count
                # print(FPS)

                self.ui.LcdNumber_1.display(str(FPS))

                img_count = 0

            # part3 识别过程 / 目标追踪

            # >>>>>>>>>>>>>>>>>神经网络接口<<<<<<<<<<<<<<<<<
            # >>>>>>>>>>>>>>>>>>>此处接入<<<<<<<<<<<<<<<<<< <
            # color_image_BGR是需要投入的图片

            detected_img, pred_boxes = yolo_detect(color_image_BGR, self.model)  # 返回检测图片和检测结果
            # 检测结果包括目标坐标，置信度 ,标签等信息

            ##>>>>>>>>>>>>>>>>>>>>>>>>接入Deep_sort<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # print("检测结果：" , pred_boxes)

            if len(pred_boxes):  # 判断结果不为空

                # Do something with my list
                show_img, result = update_tracker(pred_boxes, detected_img)

                # print("Result is:" , result)

                color_img_QImage = PySide2.QtGui.QImage(show_img, 640, 480, PySide2.QtGui.QImage.Format_RGB888)
                self.ui.label_1.setPixmap(QPixmap.fromImage(color_img_QImage))
                self.ui.label_1.setScaledContents(True)

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>显示识别结果<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                # >>>>>>>>>>>>>>>>>>>显示结果：Goal_ID = CA002;Num = 2<<<<<<<<<<<<<<<<<<<<<<<
                # self.ui.textBrowser_2.append(str(new_face))

                # 第一步：list处理，遍历元组，提取每一个目标对应元组的第5(4)个元素和第6(5)个元素
                for target in result:

                    # print("元组：" , target)
                    label = target[4]
                    id = target[5]
                    # print("label :" , label)
                    # print("id :" , id)

                    if id not in all_id:  # 追踪到新目标

                        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>结合深度数据进行二次检测<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



                        all_target.append(label)  # all_target列表存储了所有出现过的目标
                        all_id.append(id)

                    else:
                        continue

                # 第二步：统计类别及数量
                # final_result = pd.value_counts(all_target)
                final_result = Counter(all_target)
                # print(type(final_result))
                # print(final_result)

                name_list = list(final_result)

                num_list = list(final_result.values())

                # print("The object is:" , name_list)
                # print("The num is:" , num_list)

                length = len(name_list)

                if (now_time_output - last_time_output > 4):
                    last_time_output = now_time_output

                    # >>>>>>>>>>>>>>>part4 保存识别结果为txt文件<<<<<<<<<<<<<<<<
                    # >>>>>>>>>>>>>>>>>>>>>>>创建txt文件<<<<<<<<<<<<<<<<<<<<<<<
                    result_txt = open(f"./results/SICNU-SN-RD{txt_number}.txt", 'a')  # 保存文件

                    self.ui.textBrowser_1.append("运行中...")
                    self.ui.textBrowser_2.append(">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<")
                    for i_counter in range(length):
                        result_txt.write(f"Goal_ID：{name_list[i_counter]} ; Num：{num_list[i_counter]}")
                        result_txt.write('\n')

                        # print(f"目标：{name_list[i_counter]} 数量：{num_list[i_counter]}")
                        self.ui.textBrowser_2.append(f"Go al_ID：{name_list[i_counter]} ; Num：{num_list[i_counter]}")

                    pipeline.stop()
                    self.ui.textBrowser_1.append("运行结束")
                    self.ui.textBrowser_1.append(">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<")

                    # self.ui.label_1.setPixmap(QPixmap.fromImage(deep_img_QImage))
                    # self.ui.label_1.setScaledContents(True)
                    #
                    # self.ui.label_1.setPixmap(QPixmap.fromImage(color_img_QImage))
                    # self.ui.label_1.setScaledContents(True)
                    return 0


            else:  # 此帧未检测到任何目标
                # The list is empty
                color_img_QImage = PySide2.QtGui.QImage(detected_img, 640, 480, PySide2.QtGui.QImage.Format_RGB888)
                self.ui.label_1.setPixmap(QPixmap.fromImage(color_img_QImage))
                self.ui.label_1.setScaledContents(True)


#>>>>>>>>>>>>>>>>>>>>>>>>主函数<<<<<<<<<<<<<<<<<<<<<<<
if __name__ == '__main__':

    app = QApplication([])
    # stats = Stats()
    # stats.ui.show()
    # app.exec_()
    video = Stats()
    video.ui.show()
    app.exec_()