from PIL import Image, ImageFilter
import logging
from utils import ansi
import time

logger = logging.getLogger("benchmarks")

class Bench:
    name = "Pillow"

    def rotate_45_deg(img, img_name) -> float:
        logger.info(ansi(f"Using pillow to rotate {img_name} 45 degrees")) 
        img = Image.fromarray(img)

        start = time.perf_counter()
        img.rotate(45)
        end = time.perf_counter()
        delta = end - start
        logger.info(f"Time taken: {delta}")
        return delta

    def transpose(img, img_name) -> float:
        logger.info(ansi(f"Using pillow to transpose {img_name}"))
        img = Image.fromarray(img)

        start = time.perf_counter()
        img.transpose(Image.Transpose.TRANSPOSE)
        end = time.perf_counter()
        delta = end - start
        logger.info(f"Time Taken: {delta}")
        return delta

    def rgb_to_grayscale(img, img_name) -> float:
        logger.info(ansi(f"Using pillow to convert {img_name} to grayscale"))
        img = Image.fromarray(img)
        start = time.perf_counter()
        img.convert("L")
        end = time.perf_counter()
        delta = end - start
        logger.info(f"Time Taken: {delta}")
        return delta

    # https://stackoverflow.com/questions/62968174/for-pil-imagefilter-gaussianblur-how-what-kernel-is-used-and-does-the-radius-par
    def gauss_sigma_2(img, img_name) -> float:
        logger.info(ansi(f"Using pillow to convert {img_name} to grayscale"))
        img = Image.fromarray(img)
        sigma = 2
        start = time.perf_counter()
        img.filter(ImageFilter.GaussianBlur(radius=sigma))
        end = time.perf_counter()
        delta = end - start
        logger.info(f"Time Taken: {delta}")
        return delta
