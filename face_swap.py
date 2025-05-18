import cv2
import numpy as np
import insightface
import os
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import threading
import glob
from PIL import Image, ImageTk
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


class FaceSwapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("实时换脸系统")
        self.root.geometry("1280x720")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 状态变量
        self.running = False
        self.show_original = False
        self.source_face = None
        self.source_face_image = None
        self.camera_id = 0
        self.cap = None
        self.out = None
        self.save_video = False
        self.output_path = None
        self.swapper = None
        self.app = None
        self.selected_face_path = None
        self.face_thumbnails = []
        self.current_frame = None
        
        # 性能优化参数
        self.performance_mode = "平衡"  # 可选: "高质量", "平衡", "高性能"
        self.process_every_n_frames = 2  # 每N帧处理一次
        self.frame_counter = 0
        self.last_processed_frame = None
        self.det_size = (640, 640)  # 人脸检测尺寸
        self.resize_factor = 1.0  # 处理前缩放因子
        self.process_width = 640
        self.process_height = 480
        
        # 设置预选图片目录
        self.face_photos_dir = "face_photos"
        os.makedirs(self.face_photos_dir, exist_ok=True)
        
        # 初始化模型目录
        model_dir = os.path.expanduser('~/.insightface/models')
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建主界面
        self.create_widgets()
        
        # 加载预选图片
        self.load_face_thumbnails()
        
        # GPU检测
        self.has_gpu = self.check_gpu()

    def check_gpu(self):
        """检测是否可以使用GPU"""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            has_gpu = 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers
            if has_gpu:
                self.status_var.set("已检测到GPU加速支持")
            else:
                self.status_var.set("未检测到GPU加速，将使用CPU模式")
            return has_gpu
        except:
            self.status_var.set("检测GPU支持时出错，将使用CPU模式")
            return False

    def create_widgets(self):
        # 创建左右分栏
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：显示视频和控制按钮
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 视频显示区域
        self.video_canvas = tk.Canvas(left_frame, bg="black", width=640, height=480)
        self.video_canvas.pack(pady=10)
        
        # 控制按钮区域
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(controls_frame, text="摄像头ID:").pack(side=tk.LEFT, padx=5)
        self.camera_id_var = tk.StringVar(value="0")
        camera_entry = ttk.Entry(controls_frame, textvariable=self.camera_id_var, width=5)
        camera_entry.pack(side=tk.LEFT, padx=5)
        
        self.save_video_var = tk.BooleanVar(value=False)
        save_check = ttk.Checkbutton(controls_frame, text="保存视频", variable=self.save_video_var)
        save_check.pack(side=tk.LEFT, padx=15)
        
        # 性能模式选择
        perf_frame = ttk.Frame(left_frame)
        perf_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(perf_frame, text="性能模式:").pack(side=tk.LEFT, padx=5)
        self.perf_mode_var = tk.StringVar(value="平衡")
        perf_combo = ttk.Combobox(perf_frame, textvariable=self.perf_mode_var, 
                                 values=["高质量", "平衡", "高性能"], width=8, state="readonly")
        perf_combo.pack(side=tk.LEFT, padx=5)
        perf_combo.bind("<<ComboboxSelected>>", self.on_performance_change)
        
        # 添加分辨率控制
        res_frame = ttk.Frame(left_frame)
        res_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(res_frame, text="处理分辨率:").pack(side=tk.LEFT, padx=5)
        self.resolution_var = tk.StringVar(value="640x480")
        res_combo = ttk.Combobox(res_frame, textvariable=self.resolution_var, 
                               values=["320x240", "640x480", "960x720", "原始分辨率"], width=10, state="readonly")
        res_combo.pack(side=tk.LEFT, padx=5)
        
        self.start_button = ttk.Button(controls_frame, text="开始", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(controls_frame, text="停止", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.toggle_button = ttk.Button(controls_frame, text="切换显示", command=self.toggle_display, state=tk.DISABLED)
        self.toggle_button.pack(side=tk.LEFT, padx=5)
        
        # 状态显示
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(left_frame, textvariable=self.status_var, font=("Arial", 10))
        status_label.pack(pady=10)
        
        # 性能监控区域
        perf_monitor_frame = ttk.LabelFrame(left_frame, text="性能监控")
        perf_monitor_frame.pack(fill=tk.X, pady=5)
        
        self.fps_var = tk.StringVar(value="FPS: 0")
        fps_label = ttk.Label(perf_monitor_frame, textvariable=self.fps_var)
        fps_label.pack(side=tk.LEFT, padx=20)
        
        self.process_time_var = tk.StringVar(value="处理时间: 0ms")
        process_time_label = ttk.Label(perf_monitor_frame, textvariable=self.process_time_var)
        process_time_label.pack(side=tk.LEFT, padx=20)
        
        # 右侧：显示源脸图片库
        right_frame = ttk.LabelFrame(main_frame, text="预选换脸图片")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        
        # 添加新脸按钮
        add_face_frame = ttk.Frame(right_frame)
        add_face_frame.pack(fill=tk.X, pady=10)
        
        add_button = ttk.Button(add_face_frame, text="添加人脸照片", command=self.add_face_photo)
        add_button.pack(side=tk.LEFT, padx=10)
        
        refresh_button = ttk.Button(add_face_frame, text="刷新图库", command=self.load_face_thumbnails)
        refresh_button.pack(side=tk.LEFT, padx=10)
        
        # 预览当前选择的人脸
        self.preview_label = ttk.Label(right_frame, text="当前选择:")
        self.preview_label.pack(pady=(10, 5))
        
        self.preview_canvas = tk.Canvas(right_frame, width=200, height=200, bg="lightgray")
        self.preview_canvas.pack(pady=5)
        
        # 创建滚动面板来放置所有脸部缩略图
        container = ttk.Frame(right_frame)
        container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建滚动条
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建画布，并允许滚动
        self.thumbs_canvas = tk.Canvas(container, yscrollcommand=scrollbar.set)
        self.thumbs_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.thumbs_canvas.yview)
        
        # 在画布上创建一个框架来放置所有缩略图
        self.thumbs_frame = ttk.Frame(self.thumbs_canvas)
        self.thumbs_canvas.create_window((0, 0), window=self.thumbs_frame, anchor=tk.NW)
        
        self.thumbs_frame.bind("<Configure>", 
            lambda e: self.thumbs_canvas.configure(scrollregion=self.thumbs_canvas.bbox("all")))

    def on_performance_change(self, event):
        mode = self.perf_mode_var.get()
        if mode == "高质量":
            self.process_every_n_frames = 1
            self.det_size = (640, 640)
            self.resize_factor = 1.0
        elif mode == "平衡":
            self.process_every_n_frames = 2
            self.det_size = (640, 640)
            self.resize_factor = 0.75
        elif mode == "高性能":
            self.process_every_n_frames = 3
            self.det_size = (320, 320)
            self.resize_factor = 0.5
            
        self.status_var.set(f"性能模式已切换为: {mode}")

    def load_face_thumbnails(self):
        # 清除现有的缩略图
        for widget in self.thumbs_frame.winfo_children():
            widget.destroy()
            
        self.face_thumbnails = []
        
        # 获取所有照片
        face_photos = []
        for ext in ['.jpg', '.jpeg', '.png']:
            face_photos.extend(glob.glob(os.path.join(self.face_photos_dir, f"*{ext}")))
        
        if not face_photos:
            ttk.Label(self.thumbs_frame, text="没有找到照片，\n请添加人脸照片").pack(pady=20)
            return
            
        # 显示每个照片的缩略图
        for i, photo_path in enumerate(face_photos):
            frame = ttk.Frame(self.thumbs_frame)
            frame.grid(row=i//2, column=i%2, padx=5, pady=5)
            
            # 加载图片并调整大小
            try:
                img = Image.open(photo_path)
                img.thumbnail((100, 100))
                photo_img = ImageTk.PhotoImage(img)
                
                # 保存引用以防止垃圾回收
                self.face_thumbnails.append(photo_img)
                
                # 创建图像按钮
                photo_name = os.path.basename(photo_path)
                btn = tk.Button(frame, image=photo_img, 
                                command=lambda p=photo_path: self.select_face(p))
                btn.pack()
                ttk.Label(frame, text=photo_name).pack()
            except Exception as e:
                print(f"加载图片错误: {e}")
        
        # 如果有照片，默认选择第一个
        if face_photos:
            self.select_face(face_photos[0])

    def select_face(self, photo_path):
        self.selected_face_path = photo_path
        self.preview_label.config(text=f"当前选择: {os.path.basename(photo_path)}")
        
        # 更新预览
        try:
            img = Image.open(photo_path)
            img.thumbnail((200, 200))
            photo_img = ImageTk.PhotoImage(img)
            self.preview_image = photo_img  # 保存引用
            self.preview_canvas.create_image(100, 100, image=photo_img)
        except Exception as e:
            print(f"预览图片错误: {e}")
            
        # 如果相机已经运行，更新源脸
        if self.running and self.app is not None:
            self.load_source_face()

    def add_face_photo(self):
        file_path = filedialog.askopenfilename(
            title="选择人脸照片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            # 复制图片到face_photos目录
            from shutil import copy2
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.face_photos_dir, filename)
            
            try:
                copy2(file_path, dest_path)
                self.status_var.set(f"已添加照片: {filename}")
                self.load_face_thumbnails()
            except Exception as e:
                self.status_var.set(f"添加照片失败: {e}")

    def start_camera(self):
        if self.running:
            return
            
        if not self.selected_face_path:
            self.status_var.set("错误: 请先选择一个人脸照片")
            return
            
        try:
            self.camera_id = int(self.camera_id_var.get())
        except ValueError:
            self.status_var.set("错误: 请输入有效的摄像头ID")
            return
            
        self.save_video = self.save_video_var.get()
        
        # 获取选择的分辨率
        res_str = self.resolution_var.get()
        if res_str == "320x240":
            self.process_width, self.process_height = 320, 240
        elif res_str == "640x480":
            self.process_width, self.process_height = 640, 480
        elif res_str == "960x720":
            self.process_width, self.process_height = 960, 720
        else:  # 原始分辨率
            self.process_width, self.process_height = None, None
            
        # 更新性能模式设置
        self.on_performance_change(None)
            
        # 更新UI状态
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.toggle_button.config(state=tk.NORMAL)
        self.status_var.set("正在启动...")
        
        # 重置帧计数器
        self.frame_counter = 0
        self.last_processed_frame = None
        
        # 启动摄像头线程
        self.running = True
        threading.Thread(target=self.camera_thread, daemon=True).start()

    def stop_camera(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.toggle_button.config(state=tk.DISABLED)
        
        # 释放资源
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.out is not None:
            self.out.release()
            self.out = None
            
        self.status_var.set("已停止")

    def toggle_display(self):
        self.show_original = not self.show_original
        mode = "原始视频" if self.show_original else "换脸视频"
        self.status_var.set(f"显示: {mode}")

    def on_close(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()
        self.root.destroy()

    def load_source_face(self):
        try:
            self.status_var.set("正在加载源人脸...")

            # 加载源图像并提取人脸
            source_img = cv2.imread(self.selected_face_path)
            if source_img is None:
                self.status_var.set(f"无法加载源图像: {self.selected_face_path}")
                return False
                
            # 图像预处理增强
            success = self.detect_with_preprocessing(source_img)
            if success:
                return True
                
            # 如果常规检测失败，尝试使用级联分类器作为备选
            self.status_var.set("尝试使用备选人脸检测方法...")
            success = self.detect_with_cascade(source_img)
            if success:
                return True
                
            self.status_var.set("未在源图像中检测到人脸! 请尝试其他图片或调整图片质量")
            return False
        except Exception as e:
            self.status_var.set(f"加载源人脸失败: {e}")
            return False
            
    def detect_with_preprocessing(self, img):
        """使用多种图像预处理方法增强人脸检测"""
        # 尝试原始图像
        source_faces = self.app.get(img)
        if len(source_faces) > 0:
            self.source_face = source_faces[0]
            self.source_face_image = img
            self.status_var.set("人脸检测成功")
            return True
            
        # 尝试调整大小（有时更大的图像效果更好）
        if max(img.shape[0], img.shape[1]) < 800:
            scale = 2.0
            resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            source_faces = self.app.get(resized_img)
            if len(source_faces) > 0:
                self.source_face = source_faces[0]
                self.source_face_image = resized_img
                self.status_var.set("通过放大图像检测到人脸")
                return True
                
        # 尝试直方图均衡化提高对比度
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 直方图均衡化
            equalized = cv2.equalizeHist(gray)
            # 转回BGR
            enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            source_faces = self.app.get(enhanced)
            if len(source_faces) > 0:
                self.source_face = source_faces[0]
                self.source_face_image = enhanced
                self.status_var.set("通过增强对比度检测到人脸")
                return True
        except:
            pass
            
        # 尝试亮度和对比度调整
        try:
            # 提高亮度和对比度
            alpha = 1.3  # 对比度
            beta = 10    # 亮度
            adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            source_faces = self.app.get(adjusted)
            if len(source_faces) > 0:
                self.source_face = source_faces[0]
                self.source_face_image = adjusted
                self.status_var.set("通过调整亮度和对比度检测到人脸")
                return True
        except:
            pass
            
        # 检测失败
        return False
        
    def detect_with_cascade(self, img):
        """使用OpenCV级联分类器作为备选检测方法"""
        try:
            # 加载OpenCV预训练的人脸检测器
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                # 获取最大的人脸
                max_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = max_face
                
                # 为确保获取完整人脸，稍微扩大区域
                padding = int(w * 0.3)  # 30% 的填充
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(img.shape[1], x + w + padding)
                y_end = min(img.shape[0], y + h + padding)
                
                # 裁剪人脸区域
                face_img = img[y_start:y_end, x_start:x_end]
                
                # 使用insightface再次尝试检测
                source_faces = self.app.get(face_img)
                if len(source_faces) > 0:
                    self.source_face = source_faces[0]
                    self.source_face_image = face_img
                    self.status_var.set("通过级联分类器检测到人脸")
                    return True
                else:
                    # 创建一个合成的人脸对象（仅当确实需要时使用）
                    self.status_var.set("使用备选人脸检测结果")
                    
                    # 对于缺少特征的情况，可以尝试使用基本边界框
                    # 这是一个不完美的解决方案，但可以帮助处理某些边缘情况
                    try:
                        # 再次尝试在整个图像上运行insightface，但降低检测阈值
                        det_thresh_backup = self.app.det_thresh
                        self.app.det_thresh = 0.3  # 临时降低阈值
                        source_faces = self.app.get(img)
                        self.app.det_thresh = det_thresh_backup  # 恢复默认阈值
                        
                        if len(source_faces) > 0:
                            self.source_face = source_faces[0]
                            self.source_face_image = img
                            self.status_var.set("通过降低检测阈值检测到人脸")
                            return True
                    except:
                        pass
            
            return False
        except Exception as e:
            print(f"级联分类器检测失败: {e}")
            return False
            
    def init_models(self):
        try:
            self.status_var.set("正在初始化模型...")
            
            # 根据性能模式设置检测尺寸
            mode = self.perf_mode_var.get()
            if mode == "高性能":
                model_name = 'buffalo_s'  # 使用小型模型以提高性能
            else:
                model_name = 'buffalo_l'
                
            # 初始化FaceAnalysis，降低检测阈值提高检出率
            self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=self.det_size, det_thresh=0.5)  # 默认是0.6，降低以提高检出率
            
            # 检查模型文件是否存在
            model_path = 'models/inswapper_128.onnx'
            if not os.path.exists(model_path):
                alt_path = os.path.expanduser("~/.insightface/models/inswapper/inswapper_128.onnx")
                if os.path.exists(alt_path):
                    model_path = alt_path
                else:
                    self.status_var.set(f"模型文件不存在: {model_path}")
                    return False
            
            # 初始化人脸交换模型
            self.swapper = insightface.model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])
            return True
        except Exception as e:
            self.status_var.set(f"初始化模型失败: {e}")
            return False

    def preprocess_frame(self, frame):
        """预处理帧以提高性能"""
        # 根据设置调整大小
        if self.process_width and self.process_height:
            return cv2.resize(frame, (self.process_width, self.process_height))
        elif self.resize_factor != 1.0:
            return cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        return frame

    def camera_thread(self):
        # 初始化模型
        if not self.init_models():
            self.stop_camera()
            return
            
        # 加载源人脸
        if not self.load_source_face():
            self.stop_camera()
            return
            
        # 设置摄像头
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.status_var.set(f"无法打开摄像头 ID: {self.camera_id}")
            self.stop_camera()
            return

        # 设置摄像头属性，尝试降低分辨率来提高性能
        if self.perf_mode_var.get() == "高性能":
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
        # 获取摄像头属性
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30
        
        # 设置输出视频
        if self.save_video:
            output_dir = "output_videos"
            os.makedirs(output_dir, exist_ok=True)
            self.output_path = f"{output_dir}/realtime_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
        self.status_var.set(f"实时换脸系统运行中 (模式: {self.perf_mode_var.get()})")
        
        # 性能统计
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        processing_times = []
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.status_var.set("无法读取摄像头画面")
                    break

                start_time = time.time()
                
                # 保存原始帧
                original_frame = frame.copy()
                processed_frame = None
                
                # 处理FPS计算
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                    # 更新UI上的FPS显示
                    self.root.after(0, lambda: self.fps_var.set(f"FPS: {current_fps}"))
                
                # 根据帧计数器决定是否处理这一帧
                process_this_frame = (self.frame_counter % self.process_every_n_frames == 0)
                self.frame_counter += 1
                
                # 如果显示原始视频或是可以跳过处理的帧
                if self.show_original or not process_this_frame:
                    if not self.show_original and self.last_processed_frame is not None:
                        # 使用上一帧的处理结果
                        processed_frame = self.last_processed_frame
                    else:
                        processed_frame = original_frame
                else:
                    # 预处理帧以提高性能
                    small_frame = self.preprocess_frame(frame)
                    
                    # 检测目标视频帧中的人脸
                    target_faces = self.app.get(small_frame)
                    
                    if len(target_faces) > 0:
                        # 创建处理后的帧
                        processed_frame = frame.copy()

                        # 对每个检测到的人脸进行处理
                        for target_face in target_faces:
                            # 执行人脸交换
                            processed_frame = self.swapper.get(processed_frame, target_face, 
                                                          self.source_face, paste_back=True)
                            
                            # 标记出检测到的人脸
                            bbox = target_face.bbox.astype(int)
                            # 如果帧被缩放，需要调整边界框坐标
                            if self.resize_factor != 1.0 and self.process_width is None:
                                bbox = (bbox / self.resize_factor).astype(int)
                            cv2.rectangle(processed_frame, (bbox[0], bbox[1]), 
                                       (bbox[2], bbox[3]), (0, 255, 0), 1)
                    else:
                        processed_frame = original_frame
                        
                    # 保存这一帧的处理结果，供下一帧使用
                    self.last_processed_frame = processed_frame.copy()
                
                # 计算处理一帧的时间
                process_time = time.time() - start_time
                processing_times.append(process_time)
                if len(processing_times) > 30:
                    processing_times.pop(0)
                avg_process_time = sum(processing_times) / len(processing_times)
                
                # 更新UI上的处理时间显示
                self.root.after(0, lambda t=avg_process_time: 
                                self.process_time_var.set(f"处理时间: {t*1000:.1f}ms"))
                
                # 选择显示的帧
                display_frame = original_frame if self.show_original else processed_frame
                
                # 在画面上显示信息
                # 简化信息显示以减少绘制开销
                cv2.putText(display_frame, f"FPS: {current_fps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示当前模式
                mode_text = "原始" if self.show_original else "换脸"
                cv2.putText(display_frame, f"{mode_text} | {self.perf_mode_var.get()}", 
                         (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 保存视频
                if self.out is not None:
                    self.out.write(display_frame)
                
                # 转换为PIL格式以在Tkinter上显示
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(display_frame_rgb)
                pil_img = pil_img.resize((640, 480))
                tk_img = ImageTk.PhotoImage(image=pil_img)
                
                # 更新UI
                self.current_frame = tk_img  # 保存引用
                self.video_canvas.create_image(320, 240, image=tk_img)
                
            except Exception as e:
                self.status_var.set(f"处理错误: {e}")
        
        # 清理
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.out is not None:
            self.out.release()
            self.out = None
            self.status_var.set(f"视频已保存到: {self.output_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceSwapApp(root)
    root.mainloop()