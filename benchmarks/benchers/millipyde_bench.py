from utils import ansi
import logging
import time
from skimage import io
import sys
import pathlib

cur_lib_path = pathlib.Path().absolute()
sys.path.append(str(cur_lib_path))
import millipyde as mp

logger = logging.getLogger("benchmarks")


class Bench:
    name = "Millipyde"

    def rgb_to_grayscale(img, img_name) -> float:
        logger.info(ansi(f"\nGreyscaling {img_name}\n"))
        d_img = mp.gpuimage(img)

        start = time.perf_counter()
        d_img.rgb2grey()
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(delta))
        return delta

    # def rgb_to_grayscale_then_transpose(img, img_name) -> float:
    #     logger.info(ansi(f"\nGreyscaling and transposing {img_name}\n"))
    #     img_on_gpu = mp.gpuimage(img)
    #
    #     start = time.perf_counter()
    #     img_on_gpu.rgb2grey()
    #     img_on_gpu.transpose()
    #     stop = time.perf_counter()
    #     delta: float = stop - start
    #
    #     logger.info("\nTime to convert image: {}\n".format(delta))
    #     return delta

    def transpose(img, img_name) -> float:
        logger.info(ansi(f"\nTransposing {img_name}\n"))
        d_img = mp.gpuimage(img)

        start = time.perf_counter()
        d_img.transpose()
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(delta))
        return delta

    def gauss_sigma_2(img, img_name) -> float:
        logger.info(ansi(f"\nDoing Gaussian blur on {img_name}\n"))
        d_img = mp.gpuimage(img)
        d_img.rgb2grey()
        start = time.perf_counter()
        d_img.gaussian(2)
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(delta))
        return delta

    def rotate_90_deg(img, img_name) -> float:
        logger.info(ansi(f"\nRotating {img_name}\n"))
        d_img = mp.gpuimage(img)
        import math
        theta = math.radians(90)

        start = time.perf_counter()
        d_img.rotate(theta)
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(delta))
        return delta

    def adjust_gamma_2_gain_1(img, img_name) -> float:
        logger.info(ansi(f"\nAdjusting the gamma of {img_name}\n"))
        d_img = mp.gpuimage(img)
        d_img.rgb2grey()

        start = time.perf_counter()
        d_img.adjust_gamma(2, 1)
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(delta))
        return delta


def main():
    import os
    import re
    INPUT_DIR = "benchmarks/inputs/"
    input_name_re = r'charlie(\d+).png'
    # sort images (charlie{num}.png) by num
    images = sorted(os.listdir(INPUT_DIR),
                    key=lambda p: int(re.search(input_name_re, p).group(1)))
    b = Bench()
    for path in images:
        img = io.imread(INPUT_DIR + path)
        img_name = path.strip(".png")
        b.rgb_to_greyscale(img, img_name)
        b.rgb_to_grayscale_then_transpose(img, img_name)
        b.transpose(img, img_name)
        b.rotate_45_deg(img, img_name)
        b.gamma(img, img_name)
        b.gauss(img, img_name)


if __name__ == '__main__':
    main()
