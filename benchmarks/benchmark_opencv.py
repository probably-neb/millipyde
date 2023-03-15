import cv2
import utils
import numpy as np
from benchmark_scikit_image import TRUNCATE

SIGMA_2 = 2

# using the same kernel size as scipy as scipy gets the closest to millipydes output
#
# how scipy calculates the kernel radius
# https://github.com/scipy/scipy/blob/5350d42d2bafbf4d500fe874ba9af603fff012a8/scipy/ndimage/_filters.py#L269
GAUSS_KERNEL_RAD = int(SIGMA_2 * TRUNCATE + 0.5)
# https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
# getting kdim from kernel radius
GAUSS_KDIM = 2 * GAUSS_KERNEL_RAD + 1


def load_image_from_path(path: str):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    # use rgba like a normal library
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return image


def rgb_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)


def transpose(image):
    return cv2.transpose(image)


def gauss_sigma_2(image):
    return cv2.GaussianBlur(
        image, ksize=(GAUSS_KDIM, GAUSS_KDIM), sigmaX=SIGMA_2, sigmaY=SIGMA_2, borderType=cv2.BORDER_CONSTANT
    )


def compare_gauss_sigma_2(actual,millipyde):
    assert not np.all(actual[:,:,-1] == 1.0), "Assumed opencv gaussian blur did not set the alpha channel of each pixel to 1.0 but it did"
    assert np.all(millipyde[:,:,-1] == 1.0), "Assumed millipyde gaussian blur set the alpha channel of each pixel to 1.0 bit it didn't"
    # NOTE: the difference between the channels of each image is consistent
    raise utils.UnavoidableDifference(f"millipyde sets the alpha channel of every pixel to 1.0, opencv treats it as another channel. Diff between rgb channels is consistent: mean: {np.mean(actual - millipyde):.3} std dev: {np.std(actual-millipyde):.3}")

utils.create_output_verifier(gauss_sigma_2, locals(),image_to_ndarray=utils.identity, verify_output=compare_gauss_sigma_2)


def rotate_90_deg(image):
    """taken from https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py
     using this over the builtin opencv.rotate function to maintain consistent behavior with millipyde
       opencv.rotate takes a (w,h) image and turns it into a (h,w) image
       this implementation as well as the majority of other tools return an image with
       the same dimensions leaving the gaps empty

    example behavior of this implementation:

        img a = | v | after rotate = |   |
                | v |                | > |
                | v |                |   |
    """
    (h, w) = image.shape[:2]

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


utils.load_funcs(locals())
