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
from benchers.bencher_interface import BenchmarkResult
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
    imgpath = abspath(path.join(INPUTS_DIR, "charlie12.png"))
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
        paths.setdefault(result.image_name, {}).setdefault(result.benchmark, {}).update(
            {result.tool: result}
        )

    logger = get_logger().getChild("check_outputs")
    import copy

    check_results = copy.deepcopy(paths)

    for image_name, image_results in paths.items():
        for benchmark, benchmark_results in image_results.items():
            tools = check_results[image_name][benchmark].keys()
            for tool in tools:
                check_results[image_name][benchmark][tool] = None

            import itertools

            comparisons = []
            for a, b in itertools.product(tools, tools):
                if a != b:
                    ord = sorted([a, b])
                    comparisons.append((ord[0], ord[1]))
            comparisons = set(comparisons)

            # All of this 'colors' nonsense results in a dictionary
            # of values that show which outputs are similar to each other

            colors = check_results[image_name][benchmark]

            color = 1
            for a, b in comparisons:
                result_a = benchmark_results[a]
                result_b = benchmark_results[b]
                color_a = colors[a]
                color_b = colors[b]

                try:
                    result_a.assert_almost_equal(result_b, atol=TOLERANCE_PERCENTAGE)

                    # if almost_equal
                    color_a = color_a if color_a else color_b if color_b else color
                    color_b = color_b if color_b else color_a
                except AssertionError as e:
                    logger.error(
                        f"Result of [{benchmark}] differs between {a} & {b}\n{'-'*4}"
                    )
                    logger.error(f"{e}\n")
                    if color_a and not color_b:
                        color_b = color
                        color += 1
                    elif color_b and not color_a:
                        color_a = color
                        color += 1
                    elif not color_a and not color_b:
                        color_a = color
                        color_b = color + 1
                        color += 2
                colors[a] = color_a
                colors[b] = color_b

            # check_results[image_name][benchmark] = colors

    from pprint import pprint

    pprint(check_results)


def get_benchmarks_to_run():
    with open("./benchmarks_to_run.json") as f:
        config = json.load(f)["benchmarks"]
    logger = get_logger().getChild("get_benchmarks_to_run")

    def is_benchmark_enabled(benchmark):
        status = config.get(benchmark)
        keep = status is True or status is None
        if not keep:
            logger.info(f"benchmark: {benchmark} marked as False ... skipping")
        return keep

    all_benchmarks = Benchmarker._benchmarks

    return list(filter(is_benchmark_enabled, all_benchmarks))


def get_tools_to_benchmark():
    with open("./benchmarks_to_run.json") as f:
        config = json.load(f)["tools"]
    all_tools = benchers.get_tools()
    logger = get_logger().getChild("get_tools_to_benchmark")

    def is_tool_enabled(tool):
        status = config.get(tool.name)
        keep = status is True or status is None
        if not keep:
            logger.info(f"Tool: {tool.name} marked as False ... skipping")
        return keep

    return list(filter(is_tool_enabled, all_tools))


def store_results_data(results):
    """store the results of the benchmarks in csv format into
    the 'data_collection' directory with a filename based on the current time"""
    from datetime import datetime
    lines = []
    header = BenchmarkResult.csv_header()
    lines.append(header)
    for result in results:
        lines.append(result.csv())
    results_csv = "\n".join(lines)
    time = datetime.now()
    file_name = "data/%s.csv" % time.strftime(
        "%m-%d-%Y-%H:%M.csv"
    )
    with open(file_name, "w") as f:
        f.write(results_csv)


def main():
    logger = setup_logger()

    # clear outputs before running
    if SHOULD_STORE_OUTPUTS:
        try:
            shutil.rmtree(OUTPUTS_DIR)
        except FileNotFoundError:
            pass
        mkdir(OUTPUTS_DIR)

    results = []

    for image_path in get_single_image_path():
        logger.debug(f"Running benchmarks on {image_path}")

        for benchmark in get_benchmarks_to_run():
            logger.debug(f"Running benchmark: {benchmark}")

            benchmark_results = []
            for tool in get_tools_to_benchmark():
                logger.debug(
                    f"Running benchmark: {benchmark} on {image_path} with {tool.name}"
                )

                bname = tool.name

                result = Benchmarker.run_benchmark(tool, benchmark, image_path)

                # unimplemented benchmarks return None
                # and should log their own error
                if result is not None:
                    logger.info(result)
                    print(result)
                    benchmark_results.append(result)
                    results.append(result)
                    if SHOULD_STORE_OUTPUTS:
                        result.store_result_image(OUTPUTS_DIR)

    if SHOULD_CHECK_OUTPUTS:
        check_outputs(results)

    store_results_data(results)


if __name__ == "__main__":
    main()
