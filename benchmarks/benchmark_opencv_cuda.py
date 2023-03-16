import cv2
import utils
import numpy as np
from benchmark_opencv import GAUSS_KDIM

def rgb_to_grayscale(image):
    return cv2.cuda.cvtColor(image, cv2.COLOR_RGBA2GRAY)


def transpose(image):
    return cv2.cuda.transpose(image)


def load_image_from_path(path: str):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    # send image to gpu
    frame = cv2.cuda_GpuMat()
    frame.upload(image)

    return frame


def gpumat_from_np_array(ndarray):
    frame = cv2.cuda_GpuMat()
    frame.upload(ndarray)
    return frame


def gpumat_to_np_array(image):
    # get image back from gpu
    image = image.download()
    image = np.array(image)
    return image


def gauss_sigma_2(image):

    # chart mapping resulting int to opencv datatypes:
    # https://stackoverflow.com/a/39780825

    # NOTE: setting border modes results in alpha channel being blended and 
    # default values result in alpha value of 1.0 for every pixel like millipyde
    # also GAUSS_KDIM which is calculated how millipyde calculates it throws an error
    # because it is too large (GAUSS_KDIM=33, max=32). instead 31 is used to try and be
    # as close as possible
    filter = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC4, cv2.CV_8UC4, ksize=(31,31), sigma1=2, sigma2=2, rowBorderMode=cv2.BORDER_CONSTANT, columnBorderMode=cv2.BORDER_CONSTANT
    )
    return filter.apply(image)


def compare_gauss_sigma_2(actual, millipyde):
    assert not np.all(
        actual[:, :, -1] == 1.0
    ), "Assumed opencv gaussian blur did not set the alpha channel of each pixel to 1.0 but it did"
    assert np.all(
        millipyde[:, :, -1] == 1.0
    ), "Assumed millipyde gaussian blur set the alpha channel of each pixel to 1.0 bit it didn't"
    argb = actual[:, :, :3]
    brgb = millipyde[:, :, :3]
    raise utils.UnavoidableDifference(
        f"millipyde sets the alpha channel of every pixel to 1.0, opencv treats it as another channel. Diff between rgb channels is consistent: mean: {np.mean(argb - brgb):.3} std dev: {np.std(argb-brgb):.3}"
    )


def rotate_90_deg(image):
    """taken from https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py
     using this over the builtin opencv.rotate function to maintain consistent behavior with
     other tools:
       opencv.rotate takes a (w,h) image and turns it into a (h,w) image
       this implementation as well as the majority of other tools return an image with
       the same dimensions leaving the gaps empty

    example behavior of this implementation:

        img a = | v | after rotate = |   |
                | v |                | > |
                | v |                |   |
    """
    # frame.size == np.array.shape[:2]
    (w, h) = image.size()

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated = cv2.cuda.warpAffine(image, M, (w, h))

    return rotated

def fliplr(image):
    return cv2.cuda.flip(image, 1)


try:
    assert cv2.cuda.getCudaEnabledDeviceCount() > 0, "OpenCV Cuda Not Found"
    utils.create_output_verifier(
        gauss_sigma_2,
        locals(),
        image_from_ndarray=gpumat_from_np_array,
        image_to_ndarray=gpumat_to_np_array,
        verify_output=compare_gauss_sigma_2,
    )
    utils.load_funcs(
        locals(),
        image_from_ndarray=gpumat_from_np_array,
        image_to_ndarray=gpumat_to_np_array,
    )
except AssertionError:
    # don't load if OpenCV Cuda is not installed
    pass
