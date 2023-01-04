import numpy as np
from skimage import io, transform, exposure, filters
from skimage.color import rgb2gray, rgba2rgb

import pathlib
import sys

import time
cur_lib_path = pathlib.Path().absolute()
sys.path.append(str(cur_lib_path))


def greyscale_and_transpose_performance(img, img_name):
    print(
        '\033[95m' + f"\nGreyscaling and transposing {img_name} using SciKit-Image\n" + '\033[0m')

    start = time.perf_counter()
    grey_img = rgb2gray(rgb2gray(rgba2rgb(img)))
    transform.rotate(grey_img, 90)
    stop = time.perf_counter()

    print("\nTime to convert image: {}\n".format(stop - start))


def greyscale_performance(img, img_name):
    print('\033[95m' +
          f"\nGreyscaling {img_name} using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    img = rgb2gray(rgba2rgb(img))
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))


def transpose_performance(img, img_name):

    print('\033[95m' +
          f"\nTransposing {img_name} using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    img = np.transpose(img)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))


def rot_performance(img, img_name):
    print('\033[95m' +
          f"\nRotating {img_name} using SciKit-Image\n" + '\033[0m')
    start = time.perf_counter()
    img = transform.rotate(img, 45)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))


def gamma_performance(img, img_name):
    print('\033[95m' +
          f"\nAdjusting the gamma of {img_name} using SciKit-Image\n" + '\033[0m')
    img = rgb2gray(img)

    start = time.perf_counter()
    img = exposure.adjust_gamma(img, 2, 1)
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))


def gauss_performance(img, img_name):
    print('\033[95m' +
          f"\nDoing Gaussian blur on {img_name} using SciKit-Image\n" + '\033[0m')
    img = rgb2gray(img)
    start = time.perf_counter()
    img = filters.gaussian(img, sigma=2, cval=0, truncate=8, mode="constant")
    stop = time.perf_counter()
    print("\nTime to convert image: {}\n".format(stop - start))


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
