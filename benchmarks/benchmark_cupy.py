import cupy
import utils


def load_image_from_path(image_path: str):
    arr = utils.load_image_from_path(image_path)
    cupy_arr = cupy.array(arr)
    return cupy_arr


def rotate_90_deg(image):
    return cupy.rot90(image, k=1, axes=(1, 0))


def transpose(image):
    # note: this returns a view of the original image
    return cupy.transpose(image)


#
# def rgb_to_grayscale(image) -> float:
#     pass

# def gauss_sigma_2(image) -> float:
#     pass

utils.load_funcs(locals(), load_image=load_image_from_path)
