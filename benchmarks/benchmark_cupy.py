import cupy
from cupyx.scipy.ndimage import gaussian_filter
import utils


def rotate_90_deg(image):
    return cupy.rot90(image, k=1, axes=(1, 0))


def transpose(image):
    # note: this returns a view of the original image
    return cupy.transpose(image, axes=(1,0,2))

def to_ndarray(image):
    ndarray = cupy.asnumpy(image)
    return ndarray

def gauss_sigma_2(image):
    return gaussian_filter(image, sigma=2)

locals()[utils.CONVERTER_FUNC_NAME] = to_ndarray
#
# def rgb_to_grayscale(image) -> float:
#     pass

# def gauss_sigma_2(image) -> float:
#     pass


def load_image_from_path(image_path: str):
    arr = utils.load_image_from_path(image_path)
    cupy_arr = cupy.array(arr)
    return cupy_arr


utils.load_funcs(locals(), load_image=load_image_from_path)
