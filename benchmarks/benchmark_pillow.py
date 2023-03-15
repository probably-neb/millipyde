from PIL import Image, ImageFilter
import utils


def pil_image_from_ndarray(ndarray):
    return Image.fromarray(ndarray)


def to_ndarray(image):
    import numpy

    return numpy.array(image)


def rotate_90_deg(image):
    return image.rotate(90)


def transpose(image) -> float:
    return image.transpose(Image.Transpose.TRANSPOSE)


def rgb_to_grayscale(image) -> float:
    return image.convert("L")


# https://stackoverflow.com/questions/62968174/for-pil-imagefilter-gaussianblur-how-what-kernel-is-used-and-does-the-radius-par
def gauss_sigma_2(image) -> float:
    # sigma = 2
    return image.filter(ImageFilter.GaussianBlur(radius=2))


def compare_gauss_sigma_2(a, b):
    raise utils.UnavoidableDifference(
        "Pillow uses box filters to approximate a gaussian blur: https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.GaussianBlur"
    )


utils.create_output_verifier(
    gauss_sigma_2,
    locals(),
    image_from_ndarray=pil_image_from_ndarray,
    image_to_ndarray=to_ndarray,
    verify_output=compare_gauss_sigma_2,
)

locals()[utils.CONVERTER_FUNC_NAME] = to_ndarray

utils.load_funcs(locals(), image_from_ndarray=pil_image_from_ndarray)
