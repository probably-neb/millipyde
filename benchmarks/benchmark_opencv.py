import cv2
import utils
import numpy as np


def load_image_from_path(path: str):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    # use rgba like a normal library
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return image


def rgb_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)


def transpose(image):
    return cv2.transpose(image)


def gauss_sigma_2(image):
    return cv2.GaussianBlur(
        image, ksize=(0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_CONSTANT
    )


def rotate_90_deg(image):
    """taken from https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py
     using this over the builtin opencv.rotate function to maintain consistent behavior with
     other tools (namely millipyde):
       opencv.rotate takes a (w,h) image and turns it into a (h,w) image
       this implementation as well as the majority of other tools return an image with
       the same dimensions leaving the gaps empty

    example behavior of this implementation:

        img a = | v | after rotate = |   |
                | v |                | > |
                | v |                |   |
    """
    (h, w) = image.shape[:2]

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


utils.load_funcs(locals())
