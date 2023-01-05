import numpy as np
from skimage import io, transform, exposure, filters
from skimage.color import rgb2gray, rgba2rgb
import logging
from utils import ansi
import time

logger = logging.getLogger("benchmarks")


class Bench:
    name = "Scikit Image"

    # def greyscale_and_rotate_90_deg(img, img_name) -> float:
    #     logger.info(
    #         ansi(f"\nGreyscaling and rotating 90 degrees {img_name} using SciKit-Image\n"))
    #
    #     start = time.perf_counter()
    #     grey_img = rgb2gray(rgba2rgb(img))
    #     transform.rotate(grey_img, 90)
    #     stop = time.perf_counter()
    #     delta: float = stop - start
    #     logger.info("\nTime to convert image: {}\n".format(delta))
    #     return delta

    def rgb_to_grayscale(img, img_name) -> float:
        logger.info('\033[95m' +
                    f"\nGreyscaling {img_name} using SciKit-Image\n" + '\033[0m')
        start = time.perf_counter()
        # img = rgb2gray(rgba2rgb(img))
        img = rgb2gray(img)
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(stop - start))
        return delta

    def transpose(img, img_name) -> float:
        logger.info(ansi(f"\nTransposing {img_name} using SciKit-Image\n"))
        start = time.perf_counter()
        img = np.transpose(img)
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(stop - start))
        return delta

    def rotate_90_deg(img, img_name) -> float:
        logger.info(ansi(f"\nRotating {img_name} using SciKit-Image\n"))
        start = time.perf_counter()
        img = transform.rotate(img, 90)
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(stop - start))
        return delta

    def adjust_gamma_2_gain_1(img, img_name) -> float:
        logger.info(
            ansi(f"\nAdjusting the gamma of {img_name} using SciKit-Image\n"))
        img = rgb2gray(img)

        start = time.perf_counter()
        img = exposure.adjust_gamma(img, 2, 1)
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(stop - start))
        return delta

    def gauss_sigma_2(img, img_name) -> float:
        logger.info(
            ansi(f"\nDoing Gaussian blur on {img_name} using SciKit-Image\n"))
        img = rgb2gray(img)
        start = time.perf_counter()
        img = filters.gaussian(img, sigma=2, cval=0,
                               truncate=8, mode="constant")
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
        b.rgba_to_rgb_to_grayscale(img, img_name)
        b.greyscale_and_rotate_90_deg(img, img_name)
        b.transpose(img, img_name)
        b.rotate_45_deg(img, img_name)
        b.adjust_gamma_2_gain_1(img, img_name)
        b.gauss(img, img_name)


if __name__ == '__main__':
    main()
