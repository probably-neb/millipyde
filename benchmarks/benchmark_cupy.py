import cupy
from cupyx.scipy.ndimage import gaussian_filter
import utils


def rotate_90_deg(image):
    return cupy.rot90(image, k=1, axes=(1, 0))


def transpose(image):
    # note: this returns a view of the original image
    return cupy.transpose(image, axes=(1,0,2))

def gauss_sigma_2(image):
    return gaussian_filter(image, sigma=2)

#
# def rgb_to_grayscale(image) -> float:
#     pass

# def gauss_sigma_2(image) -> float:
#     pass


def cupy_array_from_ndarray(ndarray):
    cupy_arr = cupy.array(ndarray)
    return cupy_arr

def cupy_array_to_ndarray(image):
    ndarray = cupy.asnumpy(image)
    return utils.convert_image_type_to_float(ndarray)


locals()[utils.CONVERTER_FUNC_NAME] = cupy_array_to_ndarray

utils.load_funcs(locals(), image_from_ndarray=cupy_array_from_ndarray,image_to_ndarray=cupy_array_to_ndarray)
