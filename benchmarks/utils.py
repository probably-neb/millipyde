import pytest
from pathlib import Path
import os.path as path


def benchmarks_subpath(subpath):
    return path.join(BENCHMARKS_DIR, subpath)


BENCHMARKS_DIR = Path(__file__).parent

MILLIPYDE_DIR = Path(BENCHMARKS_DIR).parent

CORRECT_IMAGE_OUTPUT_DIR = benchmarks_subpath("/outputs/correct/")


def ansi(s) -> str:
    return f"\033[95m{s}\033[0m"


def benchmarks_list():
    return [
        "transpose",
        "gauss_sigma_2",
        "rotate_90_deg",
        "rgb_to_grayscale",
        "adjust_gamma_2_gain_1",
        "fliplr",
    ]


def load_image_from_path(path: str):
    import imageio.v2 as io

    # returns a np.array
    img = io.imread(path)

    return img


IMAGES = {}

def load_funcs(mod_locals, load_image=load_image_from_path):
    """loads functions from a modules locals
    wraps them in what pytest-benchmark wants for a benchmark
    and puts them back"""

    funcs = []
    for func_name in benchmarks_list():
        try:
            benchmark_func = mod_locals[func_name]
            funcs.append(benchmark_func)
        except KeyError:
            pass
    tool_name = mod_locals["__name__"].replace("benchmark_", "")

    @pytest.mark.parametrize("func", funcs)
    def benchmark_func(benchmark, func, image_path, rounds):
        def setup():
            image = IMAGES.get(tool_name, {}).get(image_path, None)
            if image is None:
                image = load_image(image_path)
            return (), {"image": image}

        benchmark.extra_info["image_path"] = image_path
        benchmark.pedantic(func, setup=setup, rounds=rounds)
        # benchmark(func, setup()[1]["image"])

    test_name = "benchmark_" + tool_name

    # add the test to the modules locals
    mod_locals[test_name] = benchmark_func


def get_correct_image_path(image_path, func_name):
    from pathlib import path

    image_path = path(image_path)
    assert image_path.is_file()
    image_path = path.join(
        CORRECT_IMAGE_OUTPUT_DIR, f"{image_path.stem}-{func_name}.npy"
    )
    return image_path
