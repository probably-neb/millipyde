import cv2 as cv
from utils import ansi
import time
import logging

logger = logging.getLogger("benchmarks")


def load_image_from_path(path):
    return cv.imread(path, cv.IMREAD_COLOR)


class Bench:
    name = "OpenCV"

    def rgb_to_grayscale(img,  img_name) -> float:
        logger.info(
            f"Using Opencv to convert {img_name} from rgba to grayscale")
        start = time.perf_counter()
        cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        end = time.perf_counter()
        delta = end - start
        logger.info(f"Time Taken: {delta}")
        return delta

    def transpose(img,  img_name) -> float:
        logger.info(
            f"Using Opencv to transpose {img_name}")
        start = time.perf_counter()
        cv.transpose(img)
        end = time.perf_counter()
        delta = end - start
        logger.info(f"Time Taken: {delta}")
        return delta

    def gauss_sigma_2(img, img_name) -> float:
        logger.info(
            f"Using Opencv to gaussian blur {img_name}")
        start = time.perf_counter()
        cv.GaussianBlur(img, ksize=(0, 0), sigmaX=2, sigmaY=2,
                        borderType=cv.BORDER_CONSTANT)
        end = time.perf_counter()
        delta = end - start
        logger.info(f"Time Taken: {delta}")
        return delta

    def rotate_90_deg(img, img_name) -> float:
        logger.info(
            f"Using Opencv to rotate {img_name} 45 degrees")
        start = time.perf_counter()
        cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        end = time.perf_counter()
        delta = end - start
        logger.info(f"Time Taken: {delta}")
        return delta
