import utils
import sys
import pathlib
import pytest


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


def getimg():
    path = "./inputs/charlie8.png"
    img = utils.load_image_from_path(path)
    image = mp.gpuimage(img)
    return (), {"image": image}


def transpose(image):
    image.transpose()
    return image


def gauss_sigma_2(image):
    image.gaussian(2)
    return image


def rotate_90_deg(image):
    image.rotate(90)
    return image


def adjust_gamma_2_gain_1(image):
    image.adjust_gamma(2, 1)
    return image


def rgb_to_grayscale(image):
    image.rgb2gray()
    return image


@pytest.mark.parametrize("func", utils.benchmarks_list())
def test_millipyde(benchmark, func):
    try:
        func = globals()[func]
    except KeyError:
        raise NotImplementedError(func)
    benchmark.pedantic(func, setup=getimg, rounds=10, warmup_rounds=1)
