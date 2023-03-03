import cv2
from .bencher_interface import Benchmarker, benchmark


assert cv2.cuda.getCudaEnabledDeviceCount() > 0, "OpenCV Wheels Not Found"


class OpenCVCudaBenchmarker(Benchmarker):
    name = "OpenCV-Cuda"

    def load_image_from_path(path: str):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

        # send image to gpu
        frame = cv2.cuda_GpuMat()
        frame.upload(image)

        return frame

    @classmethod
    def output_image_to_np_array(cls, frame, benchmark_name: str):
        # get image back from gpu
        image = frame.download()
        return super().output_image_to_np_array(image, benchmark_name)

    @benchmark
    def rgb_to_grayscale(frame):
        return cv2.cuda.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

    @benchmark
    def transpose(frame):
        return cv2.cuda.transpose(frame)

    @benchmark
    def gauss_sigma_2(frame):

        # get datatype of cuda_GpuMat:
        # frame.type()
        #
        # chart mapping resulting int to opencv datatypes:
        # https://stackoverflow.com/a/39780825

        filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC4, cv2.CV_8UC4, ksize=(0, 0), sigma1=2, sigma2=2
        )
        return filter.apply(frame)
        # return cv2.cuda.GaussianBlur(
        #     image, ksize=(0, 0), sigmaX=2, sigmaY=2, borderType=cv2.cuda.BORDER_CONSTANT
        # )

    @benchmark
    def rotate_90_deg(frame):
        """taken from https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py
         using this over the builtin opencv.rotate function to maintain consistent behavior with
         other tools:
           opencv.rotate takes a (w,h) image and turns it into a (h,w) image
           this implementation as well as the majority of other tools return an image with
           the same dimensions leaving the gaps empty

        example behavior of this implementation:

            img a = | v | after rotate = |   |
                    | v |                | > |
                    | v |                |   |
        """
        # frame.size == np.array.shape[:2]
        (h, w) = frame.size()

        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated = cv2.cuda.warpAffine(frame, M, (w, h))

        return rotated
