from utils import ansi
import logging
import time
from skimage import io
import sys
import pathlib

cur_lib_path = pathlib.Path().absolute()
sys.path.append(str(cur_lib_path))
import millipyde as mp


def greyscale_performance(img, img_name):
    logging.info(ansi(f"\nGreyscaling {img_name}\n"))
    d_img = mp.gpuimage(img)

    start = time.perf_counter()
    d_img.rgb2grey()
    stop = time.perf_counter()
    logging.info("\nTime to convert image: {}\n".format(stop - start))


def greyscale_and_transpose_performance(img, img_name):
    logging.info(ansi(f"\nGreyscaling and transposing {img_name}\n"))
    img_on_gpu = mp.gpuimage(img)

    start = time.perf_counter()
    img_on_gpu.rgb2grey()
    img_on_gpu.transpose()
    stop = time.perf_counter()

    logging.info("\nTime to convert image: {}\n".format(stop - start))


def transpose_performance(img, img_name):
    logging.info(ansi(f"\nTransposing {img_name}\n"))
    d_img = mp.gpuimage(img)

    start = time.perf_counter()
    d_img.transpose()
    stop = time.perf_counter()
    logging.info("\nTime to convert image: {}\n".format(stop - start))


def gauss_performance(img, img_name):
    logging.info(ansi(f"\nDoing Gaussian blur on {img_name}\n"))
    d_img = mp.gpuimage(img)
    d_img.rgb2grey()
    start = time.perf_counter()
    d_img.gaussian(2)
    stop = time.perf_counter()
    logging.info("\nTime to convert image: {}\n".format(stop - start))


def rot_performance(img, img_name):
    logging.info(ansi(f"\nRotating {img_name}\n"))
    d_img = mp.gpuimage(img)

    start = time.perf_counter()
    d_img.rotate(.785398)
    stop = time.perf_counter()
    logging.info("\nTime to convert image: {}\n".format(stop - start))


def gamma_performance(img, img_name):
    logging.info(ansi(f"\nAdjusting the gamma of {img_name}\n"))
    d_img = mp.gpuimage(img)
    d_img.rgb2grey()

    start = time.perf_counter()
    d_img.adjust_gamma(2, 1)
    stop = time.perf_counter()
    logging.info("\nTime to convert image: {}\n".format(stop - start))


def main():
    import os
    import re
    INPUT_DIR = "benchmarks/inputs/"
    input_name_re = r'charlie(\d+).png'
    # sort images (charlie{num}.png) by num
    images = sorted(os.listdir(INPUT_DIR),
                    key=lambda p: int(re.search(input_name_re, p).group(1)))
    for path in images:
        img = io.imread(INPUT_DIR + path)
        img_name = path.strip(".png")
        greyscale_performance(img, img_name)
        greyscale_and_transpose_performance(img, img_name)
        transpose_performance(img, img_name)
        rot_performance(img, img_name)
        gamma_performance(img, img_name)
        gauss_performance(img, img_name)


if __name__ == '__main__':
    main()
