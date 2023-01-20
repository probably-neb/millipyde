from .bencher_interface import Benchmarker, benchmark
from utils import ansi
import time
import sys
import pathlib

import math
RAD_OF_90_DEG = math.radians(90)

# this is required in order to load millipyde
# from the millipyde directory instead of
# where python normally finds modules
#
# the autopep8: on/off makes it so my autoformatter
# doesn't move the import millipyde as mp to the top
# of the file as it has to be after the sys path nonsense
#
# autopep8: off
cur_lib_path = str(pathlib.Path().absolute()).replace("benchmarks", "")
print(cur_lib_path)
sys.path.append(cur_lib_path)
# FIXME: commmented out while working on laptop
import millipyde as mp
# autopep8: on


class MillipydeBenchmark(Benchmarker):
    name = "Millipyde"

    @classmethod
    def load_image_from_path(cls, path: str):
        image = super().load_image_from_path(path)
        return mp.gpuimage(image)

    @benchmark
    def rgb_to_grayscale(image) -> float:
        image.rgb2grey()
        return image

    @benchmark
    def transpose(image) -> float:
        image.transpose()
        return image

    @benchmark
    def gauss_sigma_2(image) -> float:
        image.gaussian(2)
        return image

    @benchmark
    def rotate_90_deg(image) -> float:
        image.rotate(RAD_OF_90_DEG)
        return image

    @benchmark
    def adjust_gamma_2_gain_1(image) -> float:
        image.adjust_gamma(2, 1)
        return image
