import os
import os.path as osp
import numpy as np
import cv2
import dlib
from tqdm import tqdm
import json
from utils import files, FACIAL_LANDMARKS_IDXS, shape_to_np
from faceBlending import convex_hull, random_deform, piecewise_affine_transform, get_roi, forge, get_bounding
from color_transfer import color_transfer

PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
import pdb

class LMGenerator:
    """关键点生成器：处理单图片-单人脸
    """
    def __init__(self, path=None, detector=None, predictor=None):
        self.detector = dlib.get_frontal_face_detector() if not detector else detector
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH) if not predictor else predictor
    
    def faceDetect(self, img):
        ''' 检测人脸框，并返回置信度最高者。
        '''
        if isinstance(img, str):
            try:
                img = cv2.imread(img)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                print('ERROR read img @faceDetect')
        else:
            rgb = img
        boxes = self.detector(rgb, 1)
        if not boxes:  return None
        return boxes[0] if len(boxes)>0 else None
    
    def lmDetect(self, img, bbox=None, mode='dt+lm'):
        ''' 检测人脸关键点，假定单张图片单人脸
        param
        img: str or nparray
        mode: str，处理模式
        '''
        if type(img) not in [str, np.array]:
            raise NotImplementedError(
        'type(img) not supported: {}'.format(type(img)))
        if mode not in ['lm', 'dt+lm']:
            raise NotImplementedError(
        'mode not supported: {}'.format(mode))
        
        if isinstance(img, str):
            try:
                img = cv2.imread(img)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                print('ERROR read img @lmDetect')
        if mode == 'dt+lm':
            bbox = self.faceDetect(img)
        if not bbox:
            bbox = dlib.rectangle(0, 0, img.shape[1], img.shape[0])  # left, top, right, bottom
        
        landmarks = self.predictor(rgb, box=bbox)
        if landmarks is None:  return None
        landmarks = shape_to_np(landmarks)
        return landmarks

    def prepareWebface(self, dataPath, outPath, mode):
        ''' 处理 Webface 数据集的流程
        1. 抽帧；2. 关键点检测；3. 关键点保存
        param:
        dataPath: 数据集路径
        mode: 处理模式。'first'
        ''' 
        if not dataPath or not osp.isdir(dataPath):
            print('ERROR dataPath @prepareWebface')
        if mode not in ['first']:
            raise NotImplementedError('ERROR mode @prepareWebface')
        ids = os.listdir(dataPath)
        ids.sort()
        imgList = []
        # 每个 id 遍历
        for i in tqdm(ids):
            idPath = osp.join(dataPath, i)
            imgName = os.listdir(idPath)
            imgName.sort()
            if mode == 'first':
                relativePath =  osp.join(i, imgName[0])
                imgList.append(relativePath)
            else:
                raise NotImplementedError
        # 确认处理的部分
        print('imgList: \nlen(): {}\n5-samples: {}'.format(
            len(imgList), imgList[:5]))
        # 开始处理：人脸检测+关键点回归+保存
        lmList = []
        for path in tqdm(imgList):
            ldm = self.lmDetect(osp.join(dataPath, path))
            lmList.append(ldm.tolist())
        with open(outPath, 'w') as f:
            relaPath_lms = [i for i in zip(imgList, lmList)]
            json.dump(relaPath_lms, f)
        print('Done Preparing Webface, Mode={}'.format(mode))
    

class Blender:
    '''贴合器
    '''
    def __init__(self, ldmPath, dataPath):
        # 格式读取、转化。
        self.relativePaths, lms = [], []
        self.dataPath = dataPath
        with open(ldmPath, 'r') as f:
            relatPath_lms = json.load(f)
            for path, lm in relatPath_lms:
                self.relativePaths.append(path)
                lms.append(lm)
        self.lms = np.array(lms)  # 用于计算相似度矩阵
        N = self.lms.shape[0]
        self.lms = np.reshape(self.lms, (N, -1))
        print(self.lms.shape)  # (N, 64, 2)

    def search(self, idx, topk=100, selectNum=1):
        '''TODO:保证不重复
        '''
        pivot = self.lms[idx]
        subs = self.lms-pivot
        scores = (subs**2).sum(-1)  # l2 距离
        idxes = np.argpartition(scores, topk)[:topk]  # topK
        outs = np.random.choice(idxes, size=selectNum)  # TODO: 需要对 idx 去重
        # pdb.set_trace()
        return outs

    def blend(self, outPath, selectNum=1):
        '''关键点读取、搜索、核心、保存（命名问题很重要）
        '''
        # if not osp.isdir:
        #     os.mkdir(outPath)
        for i in tqdm(range(len(self.lms))):
            i_path = self.relativePaths[i]
            js = self.search(i, topk=100, selectNum=selectNum)
            for j in js:
                j_path = self.relativePaths[j]
                blended, label = self.core(i, j)
                i_name = '_'.join(osp.split(i_path)).rstrip('.jpg')
                j_name = '_'.join(osp.split(j_path)).rstrip('.jpg')
                name = '_'.join([i_name, j_name])  # j attack i
                # pdb.set_trace()
                cv2.imwrite(osp.join(outPath, name+'.jpg'), blended)
                cv2.imwrite(osp.join(outPath, name+'label_'+'.jpg'), label*255)

    def core(self, i, j):
        '''贴合：用 i 的背景，接纳 j 的前景
        '''
        paths = [self.relativePaths[k] for k in (i, j)]
        lms = [self.lms[i].reshape(-1,2) for k in (i, j)]
        # pdb.set_trace()
        imgs = []
        for path in paths:
            img = cv2.imread(osp.join(self.dataPath, path))
            imgs.append(img)
        
        hullMask = convex_hull(imgs[0].shape, lms[0])
        anchors, deformedAnchors = random_deform(hullMask.shape[:2], 4, 4)  # 随机变形存在问题：需要取变形结果与原图的交集。
        warped = piecewise_affine_transform(hullMask, anchors, deformedAnchors) # size (h, w, c) warped mask float64 [0, 1.0]
        # 将 warped 区域限制在人脸范围内，避免背景的影响
        warped *= (hullMask / hullMask.max())
        # 高斯模糊
        blured = cv2.GaussianBlur(warped, (5,5), 3)
        # 颜色矫正，迁移高斯模糊后blured mask区域的颜色，并对颜色纠正的blured区域人脸+原背景作为融合图片
        left, up, right, bot = get_roi(warped)  # 获取 warped 区域
        src = (imgs[0][up:bot,left:right,:]).astype(np.uint8)
        tgt = (imgs[1][up:bot,left:right,:]).astype(np.uint8)
        # pdb.set_trace()
        targetBgrT = color_transfer(src, tgt)
        targetBgr_T = imgs[1] * 1  # 开辟新内存空间
        targetBgr_T[up:bot,left:right,:] = targetBgrT  # 将色彩迁移的部分转移到原图片
        # 融合
        resultantFace = forge(imgs[0], targetBgr_T, blured)  # forged face
        # 混合边界
        resultantBounding = get_bounding(blured)
        return resultantFace, resultantBounding


if __name__ == '__main__':
    '''
    # 关键点生成
    dataset = LMGenerator()
    dataset.prepareWebface(
        dataPath='D:/BaiduNetdiskDownload/webface_align_112.tar/webface_align_112',
        outPath='D:/BaiduNetdiskDownload/webface_align_112.tar/XrayTest.txt',
        mode = 'first'
    )
    '''
    # 合成
    blender = Blender(
        'D:/BaiduNetdiskDownload/webface_align_112.tar/XrayTest.txt',
        'D:/BaiduNetdiskDownload/webface_align_112.tar/webface_align_112'
        )
    blender.blend('D:/BaiduNetdiskDownload/webface_align_112.tar/xrayBlended')