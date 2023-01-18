from skimage import transform, exposure, filters
from skimage.color import rgb2gray, rgba2rgb

from .bencher_interface import Benchmarker, benchmark


class SciKitImageBenchmarker(Benchmarker):
    name = "Scikit Image"

    # def greyscale_and_rotate_90_deg(img, img_name) -> float:
    #     grey_img = rgb2gray(rgba2rgb(img))
    #     transform.rotate(grey_img, 90)

    # NOTE: this worked with just rgb2gray(image) in scikit-image=0.18.2 however
    # it raised a warning that rgb2gray(rgba2rgb(image)) was the proper way
    # and doing only rgb2gray on rgba image would throw an error in the next
    # version. Adding the second function call about doubled this tests time
    # but I went with it since it will be the only way to do this
    # as of the next version
    @benchmark
    def rgb_to_grayscale(image) -> float:
        # return rgb2gray(image)
        return rgb2gray(rgba2rgb(image))

    # As far as I can tell SciKit-Image does not have a transpose function
    # @benchmark
    # def transpose(image):
        # return np.transpose(image)

    @benchmark
    def rotate_90_deg(image):
        return transform.rotate(image, 90)

    @benchmark
    def adjust_gamma_2_gain_1(image):
        return exposure.adjust_gamma(image, 2, 1)

    @benchmark
    def gauss_sigma_2(image):
        return filters.gaussian(image, sigma=2, cval=0, truncate=8, mode="constant")
