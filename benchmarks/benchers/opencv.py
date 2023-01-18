import cv2 as cv
import time
import logging
from .bencher_interface import Benchmarker, benchmark


class OpenCVBenchmarker(Benchmarker):
    name = "OpenCV"

    def load_image_from_path(path: str):
        return cv.imread(path, cv.IMREAD_COLOR)

    @benchmark
    def rgb_to_grayscale(image):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    @benchmark
    def transpose(image):
        return cv.transpose(image)

    @benchmark
    def gauss_sigma_2(image) -> float:
        return cv.GaussianBlur(image, ksize=(0, 0), sigmaX=2, sigmaY=2,
                        borderType=cv.BORDER_CONSTANT)

    @benchmark
    def rotate_90_deg(image) -> float:
        return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
