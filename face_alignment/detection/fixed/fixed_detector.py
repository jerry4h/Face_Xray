import os
import numpy as np
import torch

from ..core import FaceDetector


class FixedDetector(FaceDetector):
    '''This is a simple helper module that assumes the faces were detected already
        (either previously or are provided as ground truth).

        The class expects to find the bounding boxes in the same format used by
        the rest of face detectors, mainly ``list[(x1,y1,x2,y2),...]``.
        For each image the detector will search for a file with the same name and with one of the
        following extensions: .npy, .t7 or .pth

    '''

    def __init__(self, device, path_to_detector=None, verbose=False):
        super(FixedDetector, self).__init__(device, verbose)
        
        self.detected_faces = [[100, 100, 300, 300]]

        
    def detect_from_image(self, tensor_or_path):
        # Only strings supported
        
        if not isinstance(self.detected_faces, list):
            raise TypeError
        detected_faces = list(self.detected_faces)

        return detected_faces

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
