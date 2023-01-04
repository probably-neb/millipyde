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

    def greyscale_performance(img, img_name) -> float:
        logger.info(ansi(f"\nGreyscaling {img_name}\n"))
        d_img = mp.gpuimage(img)

        start = time.perf_counter()
        d_img.rgb2grey()
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(delta))
        return delta

    def greyscale_and_transpose_performance(img, img_name) -> float:
        logger.info(ansi(f"\nGreyscaling and transposing {img_name}\n"))
        img_on_gpu = mp.gpuimage(img)

        start = time.perf_counter()
        img_on_gpu.rgb2grey()
        img_on_gpu.transpose()
        stop = time.perf_counter()
        delta: float = stop - start

        logger.info("\nTime to convert image: {}\n".format(delta))
        return delta

    def transpose_performance(img, img_name) -> float:
        logger.info(ansi(f"\nTransposing {img_name}\n"))
        d_img = mp.gpuimage(img)

        start = time.perf_counter()
        d_img.transpose()
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(delta))
        return delta

    def gauss_performance(img, img_name) -> float:
        logger.info(ansi(f"\nDoing Gaussian blur on {img_name}\n"))
        d_img = mp.gpuimage(img)
        d_img.rgb2grey()
        start = time.perf_counter()
        d_img.gaussian(2)
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(delta))
        return delta

    def rot_performance(img, img_name) -> float:
        logger.info(ansi(f"\nRotating {img_name}\n"))
        d_img = mp.gpuimage(img)

        start = time.perf_counter()
        d_img.rotate(.785398)
        stop = time.perf_counter()
        delta: float = stop - start
        logger.info("\nTime to convert image: {}\n".format(delta))
        return delta

    def gamma_performance(img, img_name) -> float:
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
        b.greyscale_performance(img, img_name)
        b.greyscale_and_transpose_performance(img, img_name)
        b.transpose_performance(img, img_name)
        b.rot_performance(img, img_name)
        b.gamma_performance(img, img_name)
        b.gauss_performance(img, img_name)


if __name__ == '__main__':
    main()
