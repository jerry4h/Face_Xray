'''

Reference from @author Zhuolin Fu

'''

import argparse, sys, os
from os.path import basename, splitext
from PIL import Image
from functools import partial

from skimage.transform import PiecewiseAffineTransform, warp
import numpy as np
import cv2
import dlib
from tqdm import tqdm

from color_transfer import color_transfer
from utils import files, FACIAL_LANDMARKS_IDXS, shape_to_np

def main():
    args = get_parser()

    # source faces
    srcFaces = tqdm(files(args.srcFacePath, ['.jpg']))

    # real faces database
    # ds = image2pilBatch(files(args.faceDatabase, ['.jpg']))

    # face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shapePredictor)

    
    for i, srcFace in enumerate(srcFaces):
        # load bgr
        try:
            srcFaceBgr = cv2.imread(srcFace)
        except:
            tqdm.write(f'Fail loading: {srcFace}')
            continue

        # detect landmarks
        srcLms = get_landmarks(detector, predictor, cv2.cvtColor(srcFaceBgr, cv2.COLOR_BGR2RGB))
        if srcLms is None:
            tqdm.write(f'No face: {srcFace}')
            continue

        # find first face whose landmarks are close enough in real face database
        targetRgb = find_one_neighbor(detector, predictor, srcFace, srcLms, files(args.faceDatabase, ['.jpg']), args.threshold)
        if targetRgb is None: # if not found
            tqdm.write(f'No Match: {srcFace}')
            continue

        # if found
        # 产生凸包
        targetBgr = cv2.cvtColor(targetRgb, cv2.COLOR_RGB2BGR)
        hullMask = convex_hull(srcFaceBgr.shape, srcLms) # size (h, w, c) mask of face convex hull

        # 产生随机变形
        anchors, deformedAnchors = random_deform(hullMask.shape[:2], 4, 4)

        # 分段仿射变换
        warped = piecewise_affine_transform(hullMask, anchors, deformedAnchors) # size (h, w) warped mask

        # 高斯模糊
        blured = cv2.GaussianBlur(warped, (5,5), 3)

        # 颜色矫正，迁移高斯模糊后blured mask区域的颜色，并对颜色纠正的blured区域人脸+原背景作为融合图片
        targetBgrT = color_transfer((srcFaceBgr*blured).astype(np.uint8), (targetBgr*blured).astype(np.uint8)) + (targetBgr*(1-blured)).astype(np.uint8)

        # 融合
        resultantFace = forge(srcFaceBgr, targetBgrT, blured)  # forged face

        # 混合边界
        resultantBounding = get_bounding(blured)

        # save face images
        cv2.imwrite(f'./results/mask_{i}.jpg', hullMask)
        cv2.imwrite(f'./results/deformed_{i}.jpg', warped*255)
        cv2.imwrite(f'./results/blured_{i}.jpg', blured*255)
        cv2.imwrite(f'./results/src_{i}.jpg', srcFaceBgr)
        cv2.imwrite(f'./results/target_{i}.jpg', targetBgr)
        cv2.imwrite(f'./results/target_T_{i}.jpg', targetBgrT)
        cv2.imwrite(f'./results/forge_{i}.jpg', resultantFace)
        cv2.imwrite(f'./results/bounding_{i}.jpg', resultantBounding*255)


def get_landmarks(detector, predictor, rgb):
    # first get bounding box (dlib.rectangle class) of face.
    boxes = detector(rgb, 1)
    for box in boxes:
        landmarks = shape_to_np(predictor(rgb, box=box))
        break
    else:
        return None
    return landmarks.astype(np.int32)


def find_one_neighbor(detector, predictor, srcPath, srcLms, faceDatabase, threshold):
    for face in faceDatabase:
        rgb = dlib.load_rgb_image(face)
        landmarks = get_landmarks(detector, predictor, rgb)
        if landmarks is None:
            continue
        dist = distance(srcLms, landmarks)
        if dist < threshold and basename(face).split('_')[0] != basename(srcPath).split('_')[0]:
            return rgb
    return None


def forge(srcRgb, targetRgb, mask):

    return (mask * targetRgb + (1 - mask) * srcRgb).astype(np.uint8)

def get_bounding(mask):

    bounding = np.zeros((mask.shape[1], mask.shape[0], 3))
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            bounding[i, j] = mask[i, j] * (1 - mask[i, j]) * 4 # 处理每个像素点
    return bounding


def convex_hull(size, points, fillColor=(255,)*3):
    mask = np.zeros(size, dtype=np.uint8) # mask has the same depth as input image
    points = cv2.convexHull(np.array(points))
    corners = np.expand_dims(points, axis=0).astype(np.int32)
    cv2.fillPoly(mask, corners, fillColor)
    return mask


def random_deform(imageSize, nrows, ncols, mean=0, std=5):
    '''
    e.g. where nrows = 6, ncols = 7
    *_______*______*_____*______*______*_________*
    |                                            |
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *_______*______*_____*______*______*_________*

    '''
    h, w = imageSize
    rows = np.linspace(0, h, nrows).astype(np.int32)
    cols = np.linspace(0, w, ncols).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])
    return anchors, deformed.astype(np.int32)


def piecewise_affine_transform(image, srcAnchor, tgtAnchor):
    trans = PiecewiseAffineTransform()
    trans.estimate(srcAnchor, tgtAnchor)
    warped = warp(image, trans)
    return warped


def distance(lms1, lms2):
    return np.linalg.norm(lms1 - lms2)  # 两landmarks的二范数 = 欧几里得距离


def get_parser():
    parser = argparse.ArgumentParser(description='Demo for face x-ray fake sample generation')
    parser.add_argument('--srcFacePath', '-sfp', type=str, default='./IB.jpg')
    parser.add_argument('--faceDatabase', '-fd', type=str, default='./IF0.jpg')
    parser.add_argument('--threshold', '-t', type=float, default=250, help='threshold for facial landmarks distance')
    parser.add_argument('--shapePredictor', '-sp', type=str, default='./shape_predictor_68_face_landmarks.dat', help='Path to dlib facial landmark predictor model')
    return parser.parse_args()


if __name__ == '__main__':
    main()
