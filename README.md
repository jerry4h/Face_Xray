# Face_Xray
A 3rd-party implemented Face-Xray for deepfake detection.

## 开发说明
- [ ] 生成器 [dataset.py](dataset.py)
  - [ ] `LMGenerator`: 数据集处理接口
    - [x] `faceDetect`：dlib
    - [ ] `lmDetect`
      - [x] Dlib
      - [ ] 更换先进关键点检测器
    - [x] `prepareWebface`：高层数据集接口
      - [x] 保存关键点
  - [ ] 合成接口 `Blender`
    - [x] `__init__` 读取关键点
    - [x] `search`
      - [ ] 搜索去重
    - [x] `core`
    - [x] `blend`
      - [x] 数据保存
    - [ ] 色彩迁移改进：矩形框到自适应边框
- [ ] 判别器