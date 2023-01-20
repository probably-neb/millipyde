from os import listdir
from os.path import join, dirname, abspath
import numpy.testing as npt
import json

from typing import List
import logging

from benchers import Benchmarker
import benchers

SHOULD_CHECK_OUTPUTS = True
DECIMAL_THRESHOLD = 4
# NOTE: change this to something like logging.DEBUG
# to stop logging messages from showing up
LOG_LEVEL = logging.WARNING

INPUTS_DIR = join(dirname(abspath(__file__)), "inputs")


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
    with open('./benchmarks_to_run.json') as f:
        benchmarks_to_run = json.load(f)

    benchmarkers = benchers.get_benchmarkers()

    for image_path in get_single_image_path():
        logger.debug(
            f"Running benchmarks on {image_path}")
        for benchmark in Benchmarker._benchmarks:
            logger.debug(f"Running benchmark: {benchmark}")
            benchmark_results = []
            for benchmarker in benchmarkers:
                logger.debug(
                    f"Running benchmark: {benchmark} on {image_path} with {benchmarker.name}")
                bname = benchmarker.name
                should_run = benchmarks_to_run.get(bname)
                if not should_run:
                    if should_run is None:
                        logger.warn(
                            f"Tool: {bname} not in `benchmarks_to_run.json`... skipping")
                    else:
                        logger.info(
                            f"Tool: {bname} marked as False ... skipping")
                    continue
                result = Benchmarker.run_benchmark(
                    benchmarker, benchmark, image_path)
                # unimplemented benchmarks return None
                # and should log their own error
                if result is not None:
                    logger.info(result.__str__())
                    print(result.__str__())
                benchmark_results.append(result)
            if SHOULD_CHECK_OUTPUTS:
                logger.info("Checking outputs")
                for i in range(1, len(benchmark_results)):
                    a, b = benchmark_results[i-1], benchmark_results[i]
                    try:
                        npt.assert_almost_equal(
                            a.output_image, b.output_image, decimal=DECIMAL_THRESHOLD)
                    except AssertionError as e:
                        logger.error(
                            f"Output images for {a.benchmark} on {a.tool} and {b.tool} do not match")
                        logger.error(e)


if __name__ == "__main__":
    main()
