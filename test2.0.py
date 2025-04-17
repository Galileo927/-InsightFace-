# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import pandas as pd
import time
import os
from insightface.app import FaceAnalysis
from scipy.spatial import cKDTree
import onnxruntime as ort
import cupy as cp  # 新增 GPU 支持

class AttendanceSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("考勤打卡系统")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        self.start_button = QPushButton("考勤打卡", self)
        self.start_button.clicked.connect(self.start_attendance)
        self.layout.addWidget(self.start_button)

        self.end_button = QPushButton("结束考勤", self)
        self.end_button.clicked.connect(self.end_attendance)
        self.layout.addWidget(self.end_button)

        self.attendance_button = QPushButton("查看考勤", self)
        self.attendance_button.clicked.connect(self.view_attendance)
        self.layout.addWidget(self.attendance_button)

        self.reload_button = QPushButton("重新加载人脸库", self)
        self.reload_button.clicked.connect(self.reload_known_faces)
        self.layout.addWidget(self.reload_button)

        self.status_label = QLabel("状态：等待开始", self)
        self.layout.addWidget(self.status_label)

        self.camera = None
        self.attendance_data = pd.DataFrame(columns=["Name", "Time"])

        # 初始化 FaceAnalysis，使用 GPU
        self.app = FaceAnalysis(name='buffalo_l', root='./models')
        self.app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 表示使用 GPU

        self.known_faces = self.load_known_faces()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.confidence_threshold = 0.3
        self.current_name = None
        self.start_time = None
        self.recognition_duration = 2  # 持续识别时间阈值（秒）

        # 检查 GPU 是否启用
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print("CUDA GPU 加速启用！")
        else:
            print("警告：CUDA GPU 不可用，将使用 CPU 执行！")

    def load_known_faces(self):
        known_faces = {}
        known_embeddings = []
        known_names = []
        face_dir = r"E:\face\data\face"

        print(f"开始加载已知人脸数据，根目录：{face_dir}")

        for person_name in os.listdir(face_dir):
            person_dir = os.path.join(face_dir, person_name)
            if not os.path.isdir(person_dir):
                print(f"警告：{person_dir} 不是目录，跳过")
                continue

            embeddings = []
            for file in os.listdir(person_dir):
                file_path = os.path.join(person_dir, file)
                if not os.path.isfile(file_path):
                    print(f"警告：{file_path} 不是文件，跳过")
                    continue

                with open(file_path, 'rb') as f:
                    img_data = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"警告：无法读取文件 {file_path}，跳过")
                    continue

                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = self.app.get(img)
                    if len(faces) > 0:
                        embedding = faces[0].normed_embedding
                        # 使用 Cupy 转换到 GPU
                        embedding = cp.array(embedding)
                        embeddings.append(embedding)
                        print(f"成功生成嵌入向量：{file_path}")
                    else:
                        print(f"未检测到人脸：{file_path}")
                except Exception as e:
                    print(f"处理文件 {file_path} 时发生错误：{e}")

            if embeddings:
                # 计算平均嵌入向量
                avg_embedding = cp.mean(cp.stack(embeddings), axis=0)
                known_faces[person_name] = avg_embedding
                known_embeddings.append(avg_embedding)
                known_names.append(person_name)
                print(f"成功加载 {person_name} 的人脸数据，嵌入向量形状：{avg_embedding.shape}")
            else:
                print(f"警告：{person_name} 的目录中没有有效的照片")

        self.known_embeddings = cp.stack(known_embeddings)  # 移动到 GPU
        self.known_names = np.array(known_names)
        print("已知人脸数据加载完成")
        return known_faces

    def start_attendance(self):
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.status_label.setText("状态：无法打开摄像头，请检查设备")
                print("无法打开摄像头，请检查设备")
                return
        self.timer.start(30)  # 每 30 毫秒更新一帧
        self.status_label.setText("状态：考勤进行中")

    def update_frame(self):
        try:
            ret, frame = self.camera.read()
            if not ret:
                self.status_label.setText("状态：无法读取摄像头帧")
                self.timer.stop()
                return

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.app.get(rgb_frame)
            if faces is None:
                self.status_label.setText("状态：未检测到人脸")
                return

            detected_faces_info = []
            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                embedding = face.normed_embedding
                # 转换为 Cupy 数组并移动到 GPU
                embedding = cp.array(embedding)
                name, confidence = self.recognize_face(embedding)

                if confidence >= self.confidence_threshold:
                    detected_faces_info.append(f"{name}（置信度：{confidence:.2f}）")
                    self.draw_face_info(frame, x1, y1, x2, y2, name, confidence)
                    # 更新当前识别状态
                    if name != self.current_name:
                        self.current_name = name
                        self.start_time = time.time()  # 重置开始时间
                    else:
                        elapsed_time = time.time() - self.start_time
                        if elapsed_time >= self.recognition_duration:
                            self.update_attendance(name)
                            self.end_attendance()
                            return
                else:
                    detected_faces_info.append("Unknown（置信度：{:.2f}）".format(confidence))
                    self.draw_face_info(frame, x1, y1, x2, y2, "Unknown", confidence)

            if detected_faces_info:
                self.status_label.setText("状态：识别到人脸：" + "，".join(detected_faces_info))
            else:
                self.status_label.setText("状态：未识别到已知人脸")

            # 更新视频显示
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            self.video_label.setPixmap(QPixmap.fromImage(qImg))

        except Exception as e:
            print(f"更新帧时发生错误：{e}")
            self.status_label.setText("状态：发生错误，请检查日志")

    def draw_face_info(self, frame, x1, y1, x2, y2, name, confidence):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"置信度: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if name != "Unknown":
            cv2.putText(frame, name, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def recognize_face(self, embedding):
        # 计算与已知人脸的相似度
        distances = cp.linalg.norm(embedding - self.known_embeddings, axis=1)
        min_index = cp.argmin(distances)
        min_distance = distances[min_index].item()
        confidence = 1.0 - min_distance

        if confidence < self.confidence_threshold:
            return "Unknown", confidence
        else:
            return self.known_names[min_index], confidence

    def update_attendance(self, name):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        new_data = pd.DataFrame({"Name": [name], "Time": [current_time]})
        self.attendance_data = pd.concat([self.attendance_data, new_data], ignore_index=True)
        print(f"记录考勤：{name} - {current_time}")

    def end_attendance(self):
        if self.camera:
            self.timer.stop()
            self.camera.release()
            self.camera = None
        self.save_attendance_to_excel()
        self.attendance_data = pd.DataFrame(columns=["Name", "Time"])
        self.video_label.clear()
        self.status_label.setText("状态：考勤结束")
        print("考勤结束")

    def save_attendance_to_excel(self):
        file_name = "attendance.xlsx"
        if not os.path.exists(file_name):
            self.attendance_data.to_excel(file_name, index=False)
            print(f"考勤数据已自动保存到 {file_name}")
        else:
            existing_data = pd.read_excel(file_name)
            combined_data = pd.concat([existing_data, self.attendance_data], ignore_index=True)
            combined_data.to_excel(file_name, index=False)
            print(f"考勤数据已追加到 {file_name}")

    def view_attendance(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "打开考勤数据文件", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
        if file_name:
            attendance_data = pd.read_excel(file_name)
            print(attendance_data)
        else:
            print("未选择文件。")

    def reload_known_faces(self):
        self.known_faces = self.load_known_faces()
        self.status_label.setText("状态：已重新加载人脸库")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceSystem()
    window.show()
    sys.exit(app.exec_())