import pytest
import utils
from pathlib import Path

DEFAULT_PATH = utils.benchmarks_subpath("inputs/charlie12.png")


def pytest_addoption(parser):
    parser.addoption(
        "--images",
        nargs="*",
        default=[DEFAULT_PATH],
        help="the images to run the output verification tests on",
    )
    parser.addoption(
        "--rounds",
        dest="rounds",
        type=int,
        nargs="*",
        default=[1],
        help="the number of benchmark rounds to run",
    )
    parser.addoption(
        "--warmup-rounds",
        dest="warmup_rounds",
        type=int,
        nargs="*",
        default=[0],
        help="the number of warmup rounds to run before each benchmark",
    )
    parser.addoption(
        "--input-sizes",
        nargs="*",
        default=[1000],
        help="input sizes to run the benchmarks on. Default is 1000"
    )


def pytest_generate_tests(metafunc):
    if "image_path" in metafunc.fixturenames:
        image_params = list(
            map(
                lambda path: pytest.param(path, id=Path(path).stem),
                metafunc.config.getoption("images"),
            )
        )
        metafunc.parametrize("image_path", image_params)
    if "rounds" in metafunc.fixturenames:
        round_params = list(
            map(
                lambda r: pytest.param(r, id=f'{r}-rounds'),
                metafunc.config.getoption("rounds"),
            )
        )
        metafunc.parametrize(
                "rounds", round_params
        )
    if "warmup_rounds" in metafunc.fixturenames:
        round_params = list(
            map(
                lambda r: pytest.param(r, id=f'{r}-warmup_rounds'),
                metafunc.config.getoption("warmup_rounds"),
            )
        )
        metafunc.parametrize(
                "warmup_rounds", round_params
        )
    if "input_size" in metafunc.fixturenames:
        input_sizes = list(
            map(
                lambda r: pytest.param(int(r), id=f'{r}-input_size'),
                metafunc.config.getoption("input_sizes"),
            )
        )
        metafunc.parametrize(
                "input_size", input_sizes
        )
        
