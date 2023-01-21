from os import listdir, mkdir, path
from os.path import dirname, abspath
import shutil
import numpy.testing as npt
import json
from skimage.io import imsave
import numpy as np

from typing import List
import logging

from benchers import Benchmarker
import benchers

SHOULD_STORE_OUTPUTS = True
SHOULD_CHECK_OUTPUTS = True
TOLERANCE_PERCENTAGE = 0.05
# NOTE: change this to something like logging.DEBUG
# to stop logging messages from showing up
LOG_LEVEL = logging.WARNING

INPUTS_DIR = path.join(dirname(abspath(__file__)), "inputs")
OUTPUTS_DIR = path.join(dirname(abspath(__file__)), "outputs")

if not path.exists(OUTPUTS_DIR):
    mkdir(OUTPUTS_DIR)


def get_test_image_paths() -> List[str]:
    """
    Returns absolute paths for every image in INPUTS_DIR
    """
    images = listdir(INPUTS_DIR)
    return list(map(lambda img: abspath(path.join(INPUTS_DIR, img)), images))


def get_single_image_path() -> List[str]:
    """
    Just returns a list containing a tuple of a single image and it's name.
    Intended to reduce the time it takes to run the benchmarks
    during development
    """
    imgpath = abspath(path.join(INPUTS_DIR, "charlie10.png"))
    return [imgpath]


def setup_logger():
    logging.basicConfig(level=LOG_LEVEL)
    logger = get_logger()
    logger.setLevel(LOG_LEVEL)
    return logger


def get_logger():
    return logging.getLogger("benchmarks")


def check_outputs(results):
    # img_name: benchmark: tool: path to output image
    paths = {}

    for result in results:
        paths.setdefault(result.image_name, {}).setdefault(result.benchmark, {}).update({result.tool:result})

    logger = get_logger().getChild("check_outputs")

    for image_name, image_results in paths.items():
        for benchmark, benchmark_results in image_results.items():
            for i, (tool_a, result_a) in enumerate(benchmark_results.items()):
                for j, (tool_b, result_b) in enumerate(benchmark_results.items()):
                    # don't compare to self
                    if i != j:
                        try:
                            result_a.assert_almost_equal(result_b, atol=TOLERANCE_PERCENTAGE)
                        except AssertionError as e:
                            logger.error(f"\nResult of [{benchmark}] differs between {tool_a} & {tool_b}\n{'-'*4}")
                            logger.error(e)


def main():
    logger = setup_logger()
    with open("./benchmarks_to_run.json") as f:
        benchmarks_to_run = json.load(f)

    # clear outputs before running
    if SHOULD_STORE_OUTPUTS:
        try:
            shutil.rmtree(OUTPUTS_DIR)
        except FileNotFoundError:
            pass
        mkdir(OUTPUTS_DIR)

    benchmarkers = benchers.get_benchmarkers()
    results = []

    for image_path in get_single_image_path():
        logger.debug(f"Running benchmarks on {image_path}")

        for benchmark in Benchmarker._benchmarks:
            logger.debug(f"Running benchmark: {benchmark}")

            benchmark_results = []
            for benchmarker in benchmarkers:
                logger.debug(
                    f"Running benchmark: {benchmark} on {image_path} with {benchmarker.name}"
                )

                bname = benchmarker.name

                should_run = benchmarks_to_run.get(bname)
                if not should_run:
                    if should_run is None:
                        logger.warn(
                            f"Tool: {bname} not in `benchmarks_to_run.json`... skipping"
                        )
                    else:
                        logger.info(f"Tool: {bname} marked as False ... skipping")
                    continue

                result = Benchmarker.run_benchmark(benchmarker, benchmark, image_path)

                # unimplemented benchmarks return None
                # and should log their own error
                if result is not None:
                    logger.info(result)
                    print(result)
                    benchmark_results.append(result)
                    results.append(result)
                    if SHOULD_STORE_OUTPUTS:
                        result.store_result(OUTPUTS_DIR)

    if SHOULD_CHECK_OUTPUTS:
        check_outputs(results)
            # save_output_image(benchmarker, benchmark, image_path)
            # logger.info("Checking outputs")
            #
            # for i in range(1, len(benchmark_results)):
            #     a, b = benchmark_results[i-1], benchmark_results[i]
            #     a_img, b_img = normalize(
            #         a.output_image), normalize(b.output_image)
            #
            #     try:
            #         npt.assert_allclose(
            #             a_img, b_img, atol=TOLERANCE_PERCENTAGE)
            #
            #     except AssertionError as e:
            #         logger.error(
            #             f"Output images for {a.benchmark} on {a.tool} and {b.tool} do not match")
            #         logger.error(e)
            #
            #         benchmark_err_path = path.join(OUTPUTS_DIR, benchmark)
            #         try:
            #             mkdir(benchmark_err_path)
            #         except FileExistsError:
            #             pass
            #
            #         from skimage.util import img_as_ubyte
            #         imsave(path.join(benchmark_err_path,
            #                f"{a.tool}.png"), img_as_ubyte(a_img))
            #         imsave(path.join(benchmark_err_path,
            #                f"{b.tool}.png"), img_as_ubyte(b_img))


if __name__ == "__main__":
    main()
