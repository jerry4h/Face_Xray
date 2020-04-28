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
    - [x] Random_Deform 改进：随机缩放+向下抖动
- [ ] 判别器

## 预训练模型
百度网盘链接：https://pan.baidu.com/s/1N3NhdxzAWgHbfkcb25nUmA  提取码：uvc8

## 训练日志
- 4.20训练
  - 训练集：数据尺寸400*400，正负数据比=1：3
    - data: 15000张基于celeb数据集制作的混合人脸 + 5000张与混合人脸无重复背景的真实数据
    - label: data对应的facexray图
  - 验证集：数据尺寸400*400，正负数据比≈1：3
    - data: 5000张基于celeb数据集制作的混合人脸 + 2000张与混合人脸无重复背景的真实数据
    - label: data对应的facexray图
  - 训练相关参数及结果：百度网盘链接：https://pan.baidu.com/s/1gEs3uOP1faoimaySAhgjOw  提取码：njis
    - 目录：epoch100、200、400、500的.pth; log; tensorboard文件
  - 存在问题：loss、acc曲线表现得很正常，在epoch20左右显示已收敛，但是测试集却无法输出如同facexray的人脸轮廓图，只是类似于表明有两种噪声的图
  - 原因及解决方案
    - [ ] loss过早收敛却没学到该学的：loss太小？加大loss比例 —— loss=loss*100
