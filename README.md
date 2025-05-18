# ChangeFace - 实时换脸系统

<p align="center">
  <img src="static/img/logo.png" alt="ChangeFace Logo" width="200">
</p>

<p align="center">
  <a href="#功能特点">功能特点</a> •
  <a href="#技术架构">技术架构</a> •
  <a href="#安装使用">安装使用</a> •
  <a href="#演示效果">演示效果</a> •
  <a href="#常见问题">常见问题</a>
</p>

## 项目介绍

ChangeFace是一款基于深度学习的实时换脸系统，采用先进的人脸识别和图像处理技术，实现视频流中的实时人脸替换效果。本系统既可用于娱乐互动，也可用于工业领域的虚拟试妆、影视后期制作等场景。

系统提供图形化界面，支持多种换脸模式，具有高效、便捷的特点。Web版本让您无需复杂安装，通过浏览器即可体验实时换脸的乐趣。

## 功能特点

- **实时换脸**：摄像头视频流中的实时人脸替换
- **高质量处理**：先进的深度学习模型确保换脸效果自然
- **多种性能模式**：根据设备性能选择不同处理质量
- **人脸库管理**：内置人脸照片库，支持上传自定义照片
- **GPU加速**：支持CUDA加速，提升处理性能
- **简洁的用户界面**：直观易用的Web界面

## 技术架构

- **前端**：HTML5, CSS3, JavaScript, Bootstrap 5
- **后端**：Python, Flask
- **人脸处理**：InsightFace, OpenCV, dlib
- **深度学习框架**：ONNX Runtime

## 安装使用

### 系统要求

- Python 3.6+
- 摄像头设备
- 建议使用支持CUDA的NVIDIA显卡以获得更佳体验

### 安装步骤

1. 克隆仓库
   ```bash
   git clone https://github.com/lry888/ChangeFace.git
   cd ChangeFace
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 下载模型文件
   ```bash
   # 系统首次运行时会自动下载，或按提示手动下载
   ```

4. 运行应用
   ```bash
   python app.py
   ```

5. 在浏览器中访问
   ```
   http://localhost:5000
   ```

### 使用方法

1. 在右侧人脸库中选择一张目标人脸照片
2. 设置摄像头ID和性能模式
3. 点击"开始"按钮启动实时换脸
4. 可通过"切换显示"按钮查看原始视频流
5. 点击"停止"按钮结束处理

## 演示效果

<p align="center">
  <img src="static/img/demo.gif" alt="ChangeFace 演示" width="600">
</p>

## 常见问题

### Q: 系统报错"模型文件不存在"怎么办？
A: 需要手动下载InsightFace模型，请将inswapper_128.onnx模型文件放置在~/.insightface/models/目录下。

### Q: 换脸效果不理想怎么办？
A: 请确保光线充足，人脸清晰可见，并尝试不同的性能模式设置。

### Q: 如何提升处理帧率？
A: 选择"高性能"模式可提高帧率，但可能会略微降低质量；另外，使用支持CUDA的设备可显著提升性能。

## 贡献指南

欢迎提交Issue和Pull Request来完善本项目。提交代码前，请确保遵循以下规范：

1. 保持代码风格一致
2. 添加必要的注释说明
3. 更新相关文档

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

项目作者：[Your Name](mailto:your.email@example.com)

---

<p align="center">Copyright © 2023 ChangeFace Team</p>
