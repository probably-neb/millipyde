import cupy
from cupyx.scipy.ndimage import gaussian_filter, rotate
import numpy as np
import utils
import cv2


def cupy_array_from_ndarray(ndarray):
    cupy_arr = cupy.array(ndarray)
    return cupy_arr


def cupy_array_to_ndarray(image):
    ndarray = cupy.asnumpy(image)
    return utils.convert_image_type_to_float(ndarray)


# weights taken from https://scikit-image.org/docs/stable/api/skimage.color.html#rgb2gray
RGB2GRAY_COEFFS = cupy.array([0.2125, 0.7154, 0.0721], dtype=np.float32)


def rgb_to_grayscale(image):
    return cupy.matmul(image[:, :, :3], RGB2GRAY_COEFFS)


def f32_cupy_array_from_ndarray(ndarray):
    res = cupy_array_from_ndarray(utils.convert_image_type_to_float(ndarray)).astype(
        np.float32
    )
    assert res.dtype in [
        np.float64,
        np.float32,
    ], f"expected dtype of resulting np.array to be f32 but found {res.dtype}"
    return res


utils.create_benchmark(
    rgb_to_grayscale, locals(), image_from_ndarray=f32_cupy_array_from_ndarray
)
utils.create_output_verifier(
    rgb_to_grayscale,
    locals(),
    image_from_ndarray=f32_cupy_array_from_ndarray,
    image_to_ndarray=cupy_array_to_ndarray,
)


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


def grayscale_gauss_sigma_2(image):
    # FIXME: figure out why division by 255 is required
    return (
        gaussian_filter(
            rgb_to_grayscale(image), sigma=2, truncate=8, cval=0, mode="constant"
        )
        / 255.0
    )


utils.create_benchmark(
    grayscale_gauss_sigma_2, locals(), image_from_ndarray=f32_cupy_array_from_ndarray
)


def gauss_sigma_2(image):
    return gaussian_filter(image, sigma=2, truncate=8, cval=0, mode="constant")

def fliplr(image):
    return cupy.fliplr(image)


def compare_gauss_sigma_2(actual, millipyde):
    assert not np.all(
        actual[:, :, -1] == 1.0
    ), "Assumed cupy gaussian blur did not set the alpha channel of each pixel to 1.0 but it did"
    assert np.all(
        millipyde[:, :, -1] == 1.0
    ), "Assumed millipyde gaussian blur set the alpha channel of each pixel to 1.0 bit it didn't"
    argb = actual[:, :, :3]
    brgb = millipyde[:, :, :3]
    raise utils.UnavoidableDifference(
        f"millipyde sets the alpha channel of every pixel to 1.0, cupy treats it as another channel. Diff between rgb channels is not consistent: mean: {np.mean(argb - brgb):.3} std dev: {np.std(argb-brgb):.3}"
    )


utils.create_output_verifier(
    gauss_sigma_2,
    locals(),
    image_to_ndarray=cupy_array_to_ndarray,
    image_from_ndarray=cupy_array_from_ndarray,
    verify_output=compare_gauss_sigma_2,
)

def adjust_gamma_2_gain_1(image):
    return (((image.astype(np.float32) / 255.0) ** 2) * 255.0).astype(np.uint8)

# utils.create_benchmark(adjust_gamma_2_gain_1,locals(), image_from_ndarray=f32_cupy_array_from_ndarray)
# utils.create_output_verifier(adjust_gamma_2_gain_1,locals(), image_from_ndarray=f32_cupy_array_from_ndarray, image_to_ndarray=cupy_array_to_ndarray)

utils.load_funcs(
    locals(),
    image_from_ndarray=cupy_array_from_ndarray,
    image_to_ndarray=cupy_array_to_ndarray,
)
