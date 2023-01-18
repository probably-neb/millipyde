from os import listdir
from os.path import join, dirname, abspath
import numpy as np

from typing import List, Tuple
import logging

from benchers import Benchmarker
import benchers

# NOTE: change this to something like logging.DEBUG
# to stop logging messages from showing up
LOG_LEVEL = logging.WARNING

INPUTS_DIR = join(dirname(abspath(__file__)), "inputs")


def get_single_image_path() -> List[Tuple[str, str]]:
    """
    Just returns a list a single image.
    Intended to reduce the time it takes to run the benchmarks
    during development
    """
    path = join(INPUTS_DIR, "charlie10.png")
    return [path]


def get_test_image_paths() -> List[str]:
    """
    Returns absolute paths for every image in INPUTS_DIR
    """
    images = listdir(INPUTS_DIR)
    return list(map(lambda img: abspath(join(INPUTS_DIR, img)), images))


def get_single_image_path() -> List[str]:
    """
    Just returns a list containing a tuple of a single image and it's name.
    Intended to reduce the time it takes to run the benchmarks
    during development
    """
    path = abspath(join(INPUTS_DIR, "charlie10.png"))
    return [path]


def main():
    logging.basicConfig(level=LOG_LEVEL)
    _logger = logging.getLogger("benchmarks")
    _logger.setLevel(LOG_LEVEL)
    logger = _logger.getChild("main")

    benchmarkers = benchers.get_benchmarkers()

    for benchmark in Benchmarker._benchmarks:
        logger.debug(f"Running benchmark: {benchmark}")
        for image_path in get_single_image_path():
            logger.debug(
                f"Running benchmark: {benchmark} on {image_path}")
            for benchmarker in benchmarkers:
                logger.debug(
                    f"Running benchmark: {benchmark} on {image_path} with {benchmarker.__name__}")
                result = Benchmarker.run_benchmark(
                    benchmarker, benchmark, image_path)
                # unimplemented benchmarks return None
                # and should log their own error
                if result is not None:
                    logger.info(result.__str__())
                    print(result.__str__())


if __name__ == "__main__":
    main()
