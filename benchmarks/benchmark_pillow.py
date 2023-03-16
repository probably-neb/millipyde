from PIL import Image, ImageFilter
import utils


def pil_image_from_ndarray(ndarray):
    return Image.fromarray(ndarray)


# def to_ndarray(image):
#     import numpy
#
#     return numpy.array(image)
#


def rotate_90_deg(image):
    return image.rotate(90)


def compare_rotate_90(a, b):
    raise utils.UnavoidableDifference(
        f"{utils.percent_mismatched(a,b):.4}% mismatch due to different handling of top and bottom edge (see README)"
    )


utils.create_output_verifier(
    rotate_90_deg,
    locals(),
    image_from_ndarray=pil_image_from_ndarray,
    verify_output=compare_rotate_90,
)


def transpose(image):
    return image.transpose(Image.Transpose.TRANSPOSE)


def rgb_to_grayscale(image):
    return image.convert("L")


def gauss_sigma_2(image):
    # documentation states radius is equal to sigma
    return image.filter(ImageFilter.GaussianBlur(radius=2))


def compare_gauss_sigma_2(a, b):
    percent_mismatch = utils.percent_mismatched(a, b)
    raise utils.UnavoidableDifference(
            f"Pillow uses box filters to approximate a gaussian blur. mismatched: {percent_mismatch:.3}%"
    )


utils.create_output_verifier(
    gauss_sigma_2,
    locals(),
    image_from_ndarray=pil_image_from_ndarray,
    verify_output=compare_gauss_sigma_2,
)

def grayscale_gauss_sigma_2(image):
    return gauss_sigma_2(rgb_to_grayscale(image))

utils.create_output_verifier(
    grayscale_gauss_sigma_2,
    locals(),
    image_from_ndarray=pil_image_from_ndarray,
    verify_output=compare_gauss_sigma_2,
)


def fliplr(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


utils.load_funcs(locals(), image_from_ndarray=pil_image_from_ndarray)
