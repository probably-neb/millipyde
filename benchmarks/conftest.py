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
        help="input sizes to run the benchmarks on. Default is 1000",
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
                lambda r: pytest.param(r, id=f"{r}-rounds"),
                metafunc.config.getoption("rounds"),
            )
        )
        metafunc.parametrize("rounds", round_params)
    if "warmup_rounds" in metafunc.fixturenames:
        round_params = list(
            map(
                lambda r: pytest.param(r, id=f"{r}-warmup_rounds"),
                metafunc.config.getoption("warmup_rounds"),
            )
        )
        metafunc.parametrize("warmup_rounds", round_params)
    if "input_size" in metafunc.fixturenames:
        input_sizes = list(
            map(
                lambda r: pytest.param(int(r), id=f"{r}-input_size"),
                metafunc.config.getoption("input_sizes"),
            )
        )
        metafunc.parametrize("input_size", input_sizes)


def strip_ansi(line):
    import re

    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


def pytest_benchmark_update_json(config, benchmarks, output_json):
    import os
    import re

    ROCM_HOME = os.environ.get("ROCM_HOME", None)
    assert (
        ROCM_HOME is not None
    ), "Must export ROCM_HOME environment variable for saving Rocm info"
    import subprocess

    def cmd(argv):
        return strip_ansi(
            subprocess.run(
                argv,
                cwd=f"{ROCM_HOME}",
                capture_output=True,
                check=True,
                universal_newlines=True,
            ).stdout
        ).strip()

    gpu_info = {}
    rocm_info = {}
    tool_info = {}

    rocm_info["ROCM_HOME"] = ROCM_HOME

    ROCM_VERSION = re.search(r'(\d.\d.\d)', ROCM_HOME).groups()[0]
    rocm_info["ROCM_VERSION"] = ROCM_VERSION

    rocm_info["hipconfig"] = cmd(["hip/bin/hipconfig", "--full"])
    HIP_PLATFORM = re.search(r'HIP_PLATFORM : (\w*)', rocm_info["hipconfig"]).groups()[0]

    import cupy

    gpu = cupy.cuda.runtime.getDevice()
    gpu_info.update(cupy.cuda.runtime.getDeviceProperties(gpu))


    import importlib

    cupy_dist = ""
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if "cupy" in name:
            assert (
                cupy_dist == ""
            ), "multiple cupy versions installed. Can't tell which one is being used"
            cupy_dist = name

    tool_info["cupy"] = {
        "runtime": {
            "is_hip": cupy.cuda.runtime.is_hip,
            "driverVersion": cupy.cuda.runtime.driverGetVersion(),
        },
        "environment": {
            "cuda_path": cupy._environment.get_cuda_path(),
            "rocm_path": cupy._environment.get_rocm_path(),
            "nvcc_path": cupy._environment.get_nvcc_path(),
            "hipcc_path": cupy._environment.get_hipcc_path(),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        "version": cupy.__version__,
        "distribution": cupy_dist,
    }

    if HIP_PLATFORM == "nvidia":
        tool_info["opencv_cuda"] = {}
        import cv2
        info = cv2.cuda_DeviceInfo()
        for attr in dir(info):
            if attr[0] != "_":
                try:
                    tool_info["opencv_cuda"][attr] = getattr(info, attr)()
                except TypeError:
                    # when attr has required args
                    pass
        gpu_info["nvidia_smi"] = cmd(["nvidia-smi","-q"])


    if HIP_PLATFORM == "amd":
        import json
        gpu_info["GFX_ID"] = cmd(["bin/mygpu"])
        gpu_info.update(
            json.loads(
                cmd(
                    [
                        "bin/rocm-smi",
                        "--showproductname",
                        "--showtopo",
                        "--showdriverversion",
                        "--showmemvendor",
                        "--json",
                    ]
                )
            )
        )

        rocm_info["rocminfo"] = cmd(["bin/rocminfo"])

    output_json["machine_info"]["gpu"] = gpu_info
    output_json["rocm_info"] = rocm_info
    output_json["tool_info"] = tool_info
