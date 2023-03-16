import pytest
import utils
from skimage import transform, exposure, filters
from skimage.color import rgb2gray, rgba2rgb
from skimage.util import img_as_ubyte
import numpy as np

TRUNCATE = 8


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


def compare_rotate_90(a, b):
    raise utils.UnavoidableDifference(
        f"{utils.percent_mismatched(a,b):.4}% mismatch due to different handling of top and bottom edge (see README)"
    )


utils.create_output_verifier(rotate_90_deg, locals(), verify_output=compare_rotate_90)


def adjust_gamma_2_gain_1(image):
    return exposure.adjust_gamma(image, 2, 1)


def gauss_sigma_2(image):
    return filters.gaussian(
        image, sigma=2, cval=0, truncate=TRUNCATE, mode="constant", channel_axis=2
    )


def compare_gauss_sigma_2(actual, millipyde):
    assert not np.all(
        actual[:, :, -1] == 1.0
    ), "Assumed skimage gaussian blur did not set the alpha channel of each pixel to 1.0 but it did"
    assert np.all(
        millipyde[:, :, -1] == 1.0
    ), "Assumed millipyde gaussian blur set the alpha channel of each pixel to 1.0 bit it didn't"
    argb = actual[:, :, :3]
    brgb = millipyde[:, :, :3]
    raise utils.UnavoidableDifference(
        f"millipyde sets the alpha channel of every pixel to 1.0, skimage treats it as another channel. Diff between rgb channels is not consistent: mean: {np.mean(argb - brgb):.3} std dev: {np.std(argb-brgb):.3}"
    )


utils.create_output_verifier(
    gauss_sigma_2, locals(), #verify_output=compare_gauss_sigma_2
)


def grayscale_gauss_sigma_2(image):
    return filters.gaussian(
        rgb_to_grayscale(image), sigma=2, cval=0, truncate=8, mode="constant"
    )

def fliplr(image):
    return np.fliplr(image)

# locals()[utils.CONVERTER_FUNC_NAME] = img_as_ubyte

utils.load_funcs(locals())
