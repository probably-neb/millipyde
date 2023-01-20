from PIL import Image, ImageFilter
from .bencher_interface import Benchmarker, benchmark


class PillowBenchmarker(Benchmarker):
    name = "Pillow"

    def load_image_from_path(image_path: str):
        return Image.open(image_path)

    @benchmark
    def rotate_90_deg(image):
        return image.rotate(90)

    @benchmark
    def transpose(image) -> float:
        return image.transpose(Image.Transpose.TRANSPOSE)

    @benchmark
    def rgb_to_grayscale(image) -> float:
        return image.convert("L")

    # https://stackoverflow.com/questions/62968174/for-pil-imagefilter-gaussianblur-how-what-kernel-is-used-and-does-the-radius-par
    @benchmark
    def gauss_sigma_2(image) -> float:
        # sigma = 2
        return image.filter(ImageFilter.GaussianBlur(radius=2))
