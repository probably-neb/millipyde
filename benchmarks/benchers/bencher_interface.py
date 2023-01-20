from abc import ABC
from dataclasses import dataclass
from typing import Any

# ansi makes the output colorful
from utils import ansi
import time
import numpy as np

import collections


def repr_nparr(arr: np.ndarray):
    """A helper function to make numpy arrays print nicely"""
    return f'np.array{arr.shape}'


@dataclass
class BenchmarkResult:
    time: float
    output_image: np.ndarray
    benchmark: str
    tool: str

    def __repr__(self):
        img_str = repr_nparr(self.output_image)
        return f"BenchmarkResult(time={self.time:.5f},output_image={img_str},benchmark={self.benchmark},tool={self.tool})"


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
        parent_logger = logging.getLogger("benchmarks")
        logger = parent_logger.getChild(cls.name)

        logger.info(
            ansi(f"\nRunning benchmark {benchmark} on {image_path}\n"))

        # self refers to the class that this benchmark is in
        #
        # therefore this will use the classes own method
        # loading images or fallback on the default loader
        # in the BencherInterface class below
        image = cls.load_image_from_path(image_path)

        start = time.perf_counter()
        output_image = func(image)
        stop = time.perf_counter()
        output_image = np.array(output_image)
        # cleanup
        delta: float = stop - start
        logger.info("\nTime to run test: {}\n".format(delta))

        return BenchmarkResult(delta, output_image, benchmark, cls.name)
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
        # returns a np.array
        return io.imread(path)

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
            logger.warn(
                f"benchmark {benchmark} not implemented for {benchmarker.name}")
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
