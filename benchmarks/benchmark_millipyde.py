import utils
import sys
import pathlib
import pytest
from typing import List


# this is required in order to load millipyde
# from the millipyde directory instead of
# where python normally finds modules
#
# the autopep8: on/off makes it so my autoformatter
# doesn't move the import millipyde as mp to the top
# of the file as it has to be after the sys path nonsense
#
# autopep8: off
sys.path.append(str(utils.MILLIPYDE_DIR))
import millipyde as mp
# autopep8: on


def load_image_from_path(path: str):
    img = utils.load_image_from_path(path)
    image = mp.gpuimage(img)
    return image


def transpose(image):
    image.transpose()
    return image


def rgb_to_grayscale(image):
    image.rgb2gray()
    return image

def gauss_sigma_2(image):
    image.gaussian(2)
    return image

def grayscale_gauss_sigma_2(image):
    image.rgb2gray()
    image.gaussian(2)
    return image

def rotate_90_deg(image):
    image.rotate(90)
    return image


def adjust_gamma_2_gain_1(image):
    image.adjust_gamma(2, 1)
    return image



utils.load_funcs(locals(), load_image=load_image_from_path)

if __name__ == "__main__":
    import argparse
    import pathlib
    import numpy as np
    import imageio
    from skimage.util import img_as_float64

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        "-i",
        nargs="+",
        dest="images",
        help="the images to generate correct outputs for",
    )

    args = parser.parse_args()
    print(utils.BENCHMARKS_DIR)
    for func_name in utils.benchmarks_list():
        if not locals().get(func_name):
            continue
        func = locals()[func_name]
        for image_path in args.images:
            image = load_image_from_path(image_path)
            output_path = utils.get_correct_image_path(image_path, func_name)
            print(output_path)
            output_image = np.array(func(image))
            output_image = img_as_float64(output_image)
            print(output_image.shape, output_image.dtype)
            with open(output_path, "wb") as f:
                np.save(f, output_image)
