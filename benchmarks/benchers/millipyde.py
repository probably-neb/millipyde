from .bencher_interface import Benchmarker, benchmark
from utils import ansi
import time
# from skimage import io
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


# logger = logging.getLogger("benchmarks")


class MillipydeBenchmark(Benchmarker):
    name = "Millipyde"

    @classmethod
    def load_image_from_path(cls, path: str):
        image = super().load_image_from_path(path)
        return mp.gpuimage(image)
        # return "millipyde image"

    @benchmark
    def rgb_to_grayscale(image) -> float:
        image.rgb2grey()
        return image

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

    @benchmark
    def transpose(image) -> float:
        image.transpose()
        return image

    @benchmark
    def gauss_sigma_2(image) -> float:
        # TOODO: figure ouy why this was here?
        # d_img.rgb2grey()
        image.gaussian(2)
        return image

    @benchmark
    def rotate_90_deg(image) -> float:
        image.rotate(RAD_OF_90_DEG)
        return image

    @benchmark
    def adjust_gamma_2_gain_1(image) -> float:
        # TOODO: figure ouy why this was here?
        # d_img.rgb2grey()

        image.adjust_gamma(2, 1)
        return image
