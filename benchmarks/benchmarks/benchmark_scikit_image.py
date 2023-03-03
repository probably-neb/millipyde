import pytest
import utils
from skimage import transform, exposure, filters
from skimage.color import rgb2gray, rgba2rgb
import numpy as np

# NOTE: this worked with just rgb2gray(image) in scikit-image=0.18.2 however
# it raised a warning that rgb2gray(rgba2rgb(image)) was the proper way
# and doing only rgb2gray on rgba image would throw an error in the next
# version. Adding the second function call about doubled this tests time
# but I went with it since it will be the only way to do this
# as of the next version
def rgb_to_grayscale(image) -> float:
    # return rgb2gray(image)
    return rgb2gray(rgba2rgb(image))

def transpose(image):
    # a transpose is equivalent to 90 degree rotation ccw
    # followed by a flip along the horizantal axis
    #
    # resize is for consistency with other tools implementations
    return np.flipud(transform.rotate(image, 90, resize=True))

def rotate_90_deg(image):
    return transform.rotate(image, 90)

def adjust_gamma_2_gain_1(image):
    return exposure.adjust_gamma(image, 2, 1)

def gauss_sigma_2(image):
    return filters.gaussian(
        image, sigma=2, cval=0, truncate=8, mode="constant", channel_axis=2
    )

utils.load_funcs(locals())
