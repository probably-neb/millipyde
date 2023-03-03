import pytest
import utils

DEFAULT_PATH = utils.benchmarks_subpath("inputs/charlie1.png")


def pytest_addoption(parser):
    parser.addoption(
        "--images",
        nargs="*",
        default=[DEFAULT_PATH],
        help="the images to run the test on",
    )
    parser.addoption(
        "--benchmark-rounds",
        dest="rounds",
        type=int,
        default=5,
        help="the number of benchmark rounds to run",
    )


def pytest_generate_tests(metafunc):
    if "image_path" in metafunc.fixturenames:
        metafunc.parametrize("image_path", metafunc.config.getoption("images"))
    if "rounds" in metafunc.fixturenames:
        metafunc.parametrize("rounds", [metafunc.config.getoption("rounds")])
