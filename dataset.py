import os
import os.path as osp
import glob
import numpy as np
from random import sample, randint, uniform
import cv2
from tqdm import tqdm
import json
from utils import files, FACIAL_LANDMARKS_IDXS, shape_to_np
from faceBlending import convex_hull, random_deform, piecewise_affine_transform,\
     get_roi, forge, get_bounding, linear_deform
from color_transfer import color_transfer

PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
DETECTED_FACES = [[35, 35, 214, 214]] # [[10, 10, 246, 246]]，这个数值很关键，用Test来认真选。
SCALE, SHAKE_H =0.5, 0.2
import pdb


class DlibDetector:
    def __init__(self):
        import dlib
        self.detector = dlib.get_frontal_face_detector()

    def __call__(self, rgb, num):
        return self.detector(rgb, num)


class DlibRegressor:
    def __init__(self, modelPath):
        import dlib
        self.predictor = dlib.shape_predictor(modelPath)

    def __call__(self, rgb, bbox):
        if not isinstance(bbox, dlib.rectangle):
            bbox = dlib.rectangle(*bbox)  # left, top, right, bottom
        return self.predictor(rgb, box=bbox)


class HRRegressor:
    ''' Accurate 68-landmark Regressor
    https://github.com/1adrianb/face-alignment
    '''
    def __init__(self, device='cpu', flip_input=False, detectorType='fixed'):
        '''
        detectorType: for fixed detector, to control bbox,
         please change /face_alignment/detection/fixed/fixed_detector.py 
        '''
        import face_alignment
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,\
             device=device, flip_input=flip_input, face_detector=detectorType)
        if 'fixed' in detectorType:
            self.fa.face_detector.detected_faces = DETECTED_FACES  # 进一步封装

    def __call__(self, rgb, box=None):
        preds = self.fa.get_landmarks(rgb)
        if preds is None:  return None
        pred = preds[0]
        # import pdb
        # pdb.set_trace()
        return pred # np.uint16(pred)  #.tolist()  # 可能越界，uint8最大范围 255


class LMGenerator:
    """关键点生成器：处理单图片-单人脸
    """
    def __init__(self, detector='dlib', predictor='dlib',
     pathMode='id-name', selectMode='first', detectMode='dt+lm'):
        if pathMode not in ['name', 'id-name']:
            raise NotImplementedError('ERROR pathMode')
        self.pathMode = pathMode
        if selectMode not in ['first']:
            raise NotImplementedError('ERROR selectMode')
        self.selectMode = selectMode
        if detectMode not in ['lm', 'dt+lm']:
            raise NotImplementedError('mode not supported: {}'.format(detectMode))
        self.detectMode = detectMode

        self.detector = None
        self.predictor = None
        if 'lm' in self.detectMode:
            if predictor == 'dlib':  # todo: 解耦 Generator 与 Detector/Regressor
                self.predictor = DlibRegressor()
            elif predictor == 'hr':
                self.predictor = HRRegressor(device='cuda:0', flip_input=True)
        if 'dt' in self.detectMode:
            if detector == 'dlib':
                self.detector = DlibDetector()
    
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
    
    def lmDetect(self, img, bbox=None):
        ''' 检测人脸关键点，假定单张图片单人脸
        param
        img: str or nparray
        '''
        if type(img) not in [str, np.array]:
            raise NotImplementedError(
        'type(img) not supported: {}'.format(type(img)))
        
        if isinstance(img, str):
            try:
                img = cv2.imread(img)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                print('ERROR read img @lmDetect', img)
        if 'dt' in self.detectMode:
            bbox = self.faceDetect(img)
        if not bbox:
            bbox = [0, 0, img.shape[1], img.shape[0]]  # left, top, right, bottom
        
        landmarks = self.predictor(rgb, box=bbox)
        if landmarks is None:  return None
        landmarks = shape_to_np(landmarks)
        # import pdb
        # pdb.set_trace()
        return landmarks
    
    def prepareImgList(self, dataPath):
        if not dataPath or not osp.isdir(dataPath):
            print('ERROR dataPath:', dataPath)
        ids = os.listdir(dataPath)
        ids.sort()
        imgList = []
        # 每个 id 遍历
        if self.pathMode == 'id+name':
            for i in tqdm(ids):
                idPath = osp.join(dataPath, i)
                imgName = os.listdir(idPath)
                imgName.sort()
                if self.selectMode == 'first':
                    relativePath =  osp.join(i, imgName[0])
                    imgList.append(relativePath)
                else:
                    raise NotImplementedError
        elif self.pathMode == 'name':
            imgList = os.listdir(dataPath)
            imgList.sort()
        else:
            raise NotImplementedError
        # 确认处理的部分
        print('imgList: \nlen(): {}\n5-samples: {}'.format(
            len(imgList), imgList[:5]))
        return imgList
        
    def prepareDataset(self, dataPath, outPath):
        ''' 处理数据集的流程
        1. 抽帧；2. 关键点检测；3. 关键点保存
        param:
        dataPath: 数据集路径
        outPath: 输出的路径
        ''' 
        # imgList = self.prepareImgList(dataPath)
        imgList = glob.glob(dataPath)
        # 开始处理：人脸检测+关键点回归+保存
        lmList = []
        for path in tqdm(imgList):
            # ldm = self.lmDetect(osp.join(dataPath, path))
            # pdb.set_trace()
            ldm = self.lmDetect(path)
            if ldm is None:
                lmList.append(None)
            else:
                lmList.append(ldm.tolist())
        with open(outPath, 'w') as f:
            relaPath_lms = [i for i in zip(imgList, lmList)]
            json.dump(relaPath_lms, f)
        print('Done Preparing Dataset')


def getRelative(path):
    """
    xxx/000.mp4/0.jpg -> 000.mp4/0.jpg
    """
    path, name = osp.split(path)
    _, id = osp.split(path)
    relativePath = osp.join(id, name)
    return relativePath

def getName(path):
    """
    000.mp4/0.jpg -> 004.mp4_0
    """
    '_'.join(osp.split(path)).rstrip('.jpg')

class kernelSampler:
    """ 支持 int 类型，或者 list(int), tuple(int)
    """
    def __init__(self, kernel):
        if not isinstance(kernel, (list, tuple, int)):
            raise NotImplementedError(kernel)
        self.kernel = kernel
        if not isinstance(kernel, int):
            assert len(kernel) == 2
            self.kernel = []
            for i in range(kernel[0], kernel[1]):
                if i % 2 == 1:
                    self.kernel.append(i)

    def __call__(self):
        if isinstance(self.kernel, int):
            return self.kernel
        else:
            return sample(self.kernel, k=1)[0]


class sigmaSampler:
    """ 支持 float 类型，或者 list(a, b), tuple(a, b)
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self):
        if isinstance(self.sigma, (int, float)):
            return self.kernel
        else:
            return uniform(self.sigma[0], self.sigma[1])


class Blender:
    '''贴合器
    '''
    def __init__(self, ldmPath, dataPath, topk=100, selectNum=1, gaussianKernel=5, gaussianSigma=7):
        # 格式读取、转化。
        self.relativePaths, lms = [], []
        self.dataPath = dataPath
        self.topk = topk
        self.selectNum = selectNum
        self.gaussianKernel = gaussianKernel
        with open(ldmPath, 'r') as f:
            relatPath_lms = json.load(f)
            for path, lm in relatPath_lms:
                if lm is None:  continue
                path = getRelative(path)
                self.relativePaths.append(path)
                lms.append(lm)
        self.lms = np.array(lms)  # 用于计算相似度矩阵
        N = self.lms.shape[0]
        self.lms = np.reshape(self.lms, (N, -1))
        print(self.lms.shape)  # (N, 136)
        self.kSampler = kernelSampler(gaussianKernel)
        self.sSampler = sigmaSampler(gaussianSigma)

    def search(self, idx):
        ''' 保证不重复
        '''
        topk = min(len(self.lms)-1, self.topk)
        selectNum = min(self.selectNum, topk)
        pivot = self.lms[idx]
        subs = self.lms-pivot
        scores = (subs**2).sum(-1)  # l2 距离
        idxes = np.argpartition(scores, topk)[:topk]  # topK
        # 去重
        # 要忽略的集合
        ignoring = [idx]
        ignoring = [i for i in range(idx-100, idx+100)]  # 前后的100个都不要了
        filteredIndexes = [i for i in idxes if i not in ignoring]
        # pdb.set_trace()
        outs = sample(filteredIndexes, k=selectNum)  # 对 idx 去重
        # pdb.set_trace()
        return outs

    def blend(self, outPath):
        '''关键点读取、搜索、核心、保存（命名问题很重要）
        '''
        print('outPath: ', outPath)
        # if not osp.isdir:
        #     print('Warning: outPath not exist: ', outPath)
        #     os.mkdir(outPath)
        for i in tqdm(range(len(self.lms))):
            i_path = self.relativePaths[i]
            js = self.search(i)
            for j in js:
                # pdb.set_trace()
                j_path = self.relativePaths[j]
                blended, label = self.core(i, j)
                i_name = '_'.join(osp.split(i_path)).rstrip('.jpg')
                j_name = '_'.join(osp.split(j_path)).rstrip('.jpg')
                name = '_'.join([i_name, j_name])  # j attack i
                # pdb.set_trace()
                status = 0
                status += cv2.imwrite(osp.join(outPath, name+'.jpg'), blended)
                status += cv2.imwrite(osp.join(outPath, name+'_label'+'.jpg'), label*255)
                if status != 2:
                    print('Error: image saving failed: ', name)

    def core(self, i, j):
        '''贴合：用 i 的背景，接纳 j 的前景
        '''
        paths = [self.relativePaths[k] for k in (i, j)]
        lms = [self.lms[i].reshape(-1,2) for k in (i, j)]
        # pdb.set_trace()
        imgPair = []
        for path in paths:
            imgPath = osp.join(self.dataPath, path)
            # imgPath = path
            img = cv2.imread(imgPath)
            if img is None:
                print('Error imgPath: ', imgPath)
            imgPair.append(img)
        
        hullMask = convex_hull(imgPair[0].shape, lms[0])  # todo: shrink mask.
        # 只对 mask 部分 random deform
        left, up, right, bot = get_roi(hullMask)
        left, up, right, bot = (left+0)//2, (up+0)//2, (right+hullMask.shape[1])//2, (bot+hullMask.shape[0])//2
        centerHullMask = hullMask[up:bot, left:right, :]
        anchors, deformedAnchors = random_deform(centerHullMask.shape[:2], 4, 4)  # todo 方法不够理想
        warpedMask = piecewise_affine_transform(centerHullMask, anchors, deformedAnchors)
        # 伪造区域随机化：进一步缩放+平移抖动
        warpedMask = linear_deform(warpedMask, scale=SCALE, shake_h=SHAKE_H, random=True)
        # 将 warped 区域限制在人脸范围内，避免背景的影响
        warpedMask *= (centerHullMask / centerHullMask.max())
        # 还原
        warped = np.zeros_like(hullMask, dtype=warpedMask.dtype)
        warped[up:bot, left:right, :] = warpedMask
        # pdb.set_trace()

        # 高斯模糊
        # blured = cv2.GaussianBlur(warped, (self.gaussianKernel, self.gaussianKernel), 3)
        ksize, sigma = self.kSampler(), self.sSampler()
        # print(ksize, sigma)
        blured = cv2.GaussianBlur(warped, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        # 颜色矫正，迁移高斯模糊后blured mask区域的颜色，并对颜色纠正的blured区域人脸+原背景作为融合图片
        left, up, right, bot = get_roi(blured)  # 获取 warped 区域
        
        # src 取中间部分采集色彩（效果感觉差了很多，还是得改基于 mask 的色彩迁移）
        # h, w = bot-up, right-left
        # src = (imgPair[0][up+h//4:bot-h//4,left+w//4:right-w//4,:]).astype(np.uint8)
        src = (imgPair[0][up:bot,left:right,:]).astype(np.uint8)
        tgt = (imgPair[1][up:bot,left:right,:]).astype(np.uint8)
        # 基于 mask 的色彩迁移
        targetBgrT = color_transfer(src, tgt, preserve_paper=False, mask=blured[up:bot,left:right,0]!=0)
        # pdb.set_trace()
        targetBgr_T = imgPair[1] * 1  # 开辟新内存空间
        targetBgr_T[up:bot,left:right,:] = targetBgrT  # 将色彩迁移的部分转移到原图片
        # 融合
        resultantFace = forge(imgPair[0], targetBgr_T, blured)  # forged face
        # 混合边界
        resultantBounding = get_bounding(blured)
        return resultantFace, resultantBounding


if __name__ == '__main__':
    
    # 关键点生成
    
    # hr = HRRegressor()
    '''
    dataset = LMGenerator(detector=None, predictor='hr', pathMode='name', selectMode='first', detectMode='lm')
    dataset.prepareDataset(
        dataPath='D:/Dataset/randomSelected',
        outPath='D:/Dataset/randomSelected10.txt'
    )
    '''
    '''
    # 合成
    blender = Blender(
        'D:/Dataset/randomSelected10.txt',
        'D:/Dataset/randomSelected'
        )
    blender.blend('D:/Dataset/randomSelectedBlend10')
    '''
    '''
    dataset = LMGenerator(detector=None, predictor='hr', pathMode='name', selectMode='first', detectMode='lm')
    dataset.prepareDataset(
        dataPath='/nas/hjr/FF++c23/original/generator/*/*.jpg',
        outPath='/nas/hjr/FF++c23/original/originalC23X100kLm.txt'
    )
    '''
    
    blender = Blender(
        ldmPath='/mnt/hjr/FF++c23/original/originalC23X100kLm.txt',
        dataPath='/mnt/hjr/FF++c23/original/generator',
        topk=100, selectNum=1, gaussianKernel=[31,63], gaussianSigma=[7, 15]
        )
    blender.blend(outPath='/mnt/hjr/FF++c23/original/generatorBlendedRandomGaussian')
    
