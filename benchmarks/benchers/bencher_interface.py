from abc import ABC
from dataclasses import dataclass, field

import time
import numpy as np
from os import path
from pathlib import Path


def repr_nparr(arr: np.ndarray):
    """A helper function to make numpy arrays print nicely"""
    return f"np.array{arr.shape}"


@dataclass
class BenchmarkResult:
    time: float
    # don't print the output image
    output_image: np.ndarray = field(repr=False)
    # print this instead
    output: str = field(default="", init=False)
    dtype: str = field(default=None, init=False)
    benchmark: str
    tool: str
    # don't print full image_path
    image_path: str = field(repr=False)
    # print this instead
    input_name: str = field(default="", init=False)

    def __post_init__(self):
        self.output = repr_nparr(self.output_image)
        self.input_name = self.image_name
        self.dtype = str(self.output_image.dtype)

    @property
    def image_name(self):
        return path.basename(self.image_path).replace(".", "_")

    def store_result(self, dir: str):
        from skimage.io import imsave

        output_path = Path(dir, self.image_name, self.benchmark, f"{self.tool}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imsave(output_path, self.output_image)


def benchmark(func):
    """A function designed to be a decorator around all benchmarks"""

    benchmark = func.__name__
    # register this benchmark in the list of bencharks
    # within the Benchmarker class
    if benchmark not in Benchmarker._benchmarks:
        Benchmarker._benchmarks.append(benchmark)

    @classmethod
    def inner(cls, image_path):
        # get the right logger
        import logging

        # ansi makes the output colorful
        from utils import ansi

        parent_logger = logging.getLogger("benchmarks")
        logger = parent_logger.getChild(cls.name)

        logger.debug(ansi(f"\nRunning benchmark {benchmark} on {image_path}\n"))

        # self refers to the class that this benchmark is in
        #
        # therefore this will use the classes own method
        # loading images or fallback on the default loader
        # in the BencherInterface class below
        image = cls.load_image_from_path(image_path)

        start = time.perf_counter()
        output_image = func(image)
        stop = time.perf_counter()

        output_image = cls.output_image_to_np_array(output_image, benchmark)
        # cleanup
        delta: float = stop - start
        logger.debug("\nTime to run test: {}\n".format(delta))

        return BenchmarkResult(
            time=delta,
            output_image=output_image,
            benchmark=benchmark,
            tool=cls.name,
            image_path=image_path,
        )

    return inner


class Benchmarker(ABC):
    """
    The interface for implementing a benchmark.
    Any class in a module in the `milllipyde/benchmarks/benchers/`
    directory that has this as a superclass will have any function
    with the `@benchmark` decorator ran as a benchmark
    """

    # all subclasses should declare a name
    name = "No Name Specified"
    # this will have no effect if it is set in a subclass
    _benchmarks = []

    @classmethod
    def load_image_from_path(cls, path: str) -> np.ndarray:
        """The default way to load an image.
        Subclasses can implement this themselves
        to load images their own way without affecting
        their benchmark times"""
        from skimage import io
        from skimage.util import img_as_float

        # returns a np.array
        img = io.imread(path)
        # img = img_as_float(img)
        return img

    @classmethod
    def output_image_to_np_array(cls, output_image, benchmark_name: str) -> np.array:
        """
        This function is for reconciling the different ways different tools
        handle images such as using rgb vs rgba or floats vs integers
        """
        # from skimage.util import img_as_float
        output_image = np.array(output_image)
        # output_image = img_as_float(output_image)
        return output_image

    def run_benchmark(benchmarker, benchmark: str, image_path: str):
        """How all Benchmarks _should_ be ran

        Args:
            benchmarker: a subclass of Benchmarker
            benchmark: the stringified name of the function (i.e. "rgb_to_grayscale")
            image_path: absolute path of the image

        Returns:
            the output of the benchmark
        """
        benchmark_func = getattr(benchmarker, benchmark)
        output = benchmark_func(image_path)
        if output is None:
            import logging

            logger = logging.getLogger("benchmarks")
            logger.warn(f"benchmark {benchmark} not implemented for {benchmarker.name}")
        return output

    def rgb_to_grayscale(image):
        pass

    def transpose(image):
        pass

    def gauss_sigma_2(image):
        pass

    def rotate_90_deg(image):
        pass

    def adjust_gamma_2_gain_1(image):
        pass
