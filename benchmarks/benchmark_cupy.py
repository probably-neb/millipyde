import cupy
from cupyx.scipy.ndimage import gaussian_filter, rotate
import utils
import cv2


def cupy_array_from_ndarray(ndarray):
    cupy_arr = cupy.array(ndarray)
    return cupy_arr


def cupy_array_to_ndarray(image):
    ndarray = cupy.asnumpy(image)
    return utils.convert_image_type_to_float(ndarray)


def rotate_90_deg(image):
    return rotate(image, angle=90, axes=(1, 0), reshape=False)


def compare_rotate_90(a, b):
    raise utils.UnavoidableDifference(
        f"{utils.percent_mismatched(a,b):.4}% mismatch due to different handling of top and bottom edge (see README)"
    )


utils.create_output_verifier(
    rotate_90_deg,
    locals(),
    image_to_ndarray=cupy_array_to_ndarray,
    image_from_ndarray=cupy_array_from_ndarray,
    verify_output=compare_rotate_90,
)


def transpose(image):
    # note: due to the way cupy stores the array, this essentially just swaps a few pointers
    # i.e. it does very little work and happens incredibly fast
    return cupy.transpose(image, axes=(1, 0, 2))


def gauss_sigma_2(image):
    return gaussian_filter(
        image, sigma=2, truncate=8, cval=0, mode="constant"
    )


#
# def rgb_to_grayscale(image) -> float:
#     pass

# def gauss_sigma_2(image) -> float:
#     pass


locals()[utils.CONVERTER_FUNC_NAME] = cupy_array_to_ndarray

utils.load_funcs(
    locals(),
    image_from_ndarray=cupy_array_from_ndarray,
    image_to_ndarray=cupy_array_to_ndarray,
)
