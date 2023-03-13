from PIL import Image, ImageFilter
import utils


def load_image_from_path(image_path: str):
    return Image.open(image_path).convert("RGBA")

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

def to_ndarray(image):
    import numpy
    return numpy.array(image)

locals()[utils.CONVERTER_FUNC_NAME] = to_ndarray

utils.load_funcs(locals(), load_image=load_image_from_path)
