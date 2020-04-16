# Face_Xray
A 3rd-party implemented Face-Xray for deepfake detection.

## 开发说明
- [x] 生成器 [dataset.py](dataset.py)
  - [x] 人脸检测器封装 `xxDetector`
    - [x] Dlib 
  - [x] 关键点回归器封装 `xxRegressor`
    - [x] Dlib
    - [x] HR
      - [x] fixed 类型人脸框全局赋值 
  - [x] 生成关键点功能类 `LMGenerator`
    - [ ] 将 Detector/Regressor 功能与 Generator 解耦
  - [x] 贴合功能类 `Blender`
    - [x] color_transfer: 基于mask迁移 
- [ ] 判别器

## 预训练模型
百度网盘链接：https://pan.baidu.com/s/1N3NhdxzAWgHbfkcb25nUmA  提取码：uvc8
