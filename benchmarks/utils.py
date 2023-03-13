import pytest
from pathlib import Path
import os.path as path
import numpy as np
import numpy.testing as npt


BENCHMARKS_DIR = Path(__file__).parent

MILLIPYDE_DIR = Path(BENCHMARKS_DIR).parent


def benchmarks_subpath(subpath):
    return path.join(BENCHMARKS_DIR, subpath)


CORRECT_IMAGE_OUTPUT_DIR = benchmarks_subpath("outputs/correct/")

CONVERTER_FUNC_NAME = "convert_output_to_ndarray"

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


def get_correct_image_path(image_path, func_name):
    image_path = Path(image_path)
    if not image_path.is_file():
        image_path = BENCHMARKS_DIR / "inputs" / image_path
    assert image_path.is_file(), f"image: {image_path} does not exist"
    image_path = path.join(
        CORRECT_IMAGE_OUTPUT_DIR, f"{image_path.stem}-{func_name}.npy"
    )
    return image_path


def load_funcs(mod_locals, load_image=load_image_from_path):
    """loads functions from a modules locals
    wraps them in what pytest-benchmark wants for a benchmark
    and puts them back"""

    def get_func_from_locals(func_name, default=None):
        try:
            return mod_locals[func_name]
        except KeyError:
            return default

    funcs = []
    for func_name in benchmarks_list():
        if func_name in mod_locals:
            mod_func = mod_locals[func_name]
            funcs.append(mod_func)

    benchmark_name = mod_locals["__name__"]
    tool_name = mod_locals["__name__"].replace("benchmark_", "")
    verify_name = f"test_{tool_name}_outputs"

    def create_setup_func(image_path):
        def setup():
            image = IMAGES.get(tool_name, {}).get(image_path, None)
            if image is None:
                image = load_image(image_path)
            # this wierd return signature is required by benchmark.pedantic
            # it represents *args, **kwargs
            return (), {"image": image}

        return setup

    @pytest.mark.parametrize("func", funcs)
    def benchmark_func(benchmark, func, image_path, rounds):
        setup = create_setup_func(image_path)
        benchmark.extra_info["image_path"] = image_path
        # TODO: run file once here to get output image type?
        benchmark.pedantic(func, setup=setup, rounds=rounds)

    @pytest.mark.parametrize("func", funcs)
    def test_ouput_correct(func, image_path):
        correct_output_path = get_correct_image_path(image_path, func.__name__)
        correct_output = np.load(correct_output_path)
        setup = create_setup_func(image_path)
        input_image = setup()[1]["image"]
        this_output = func(input_image)

        
        if not isinstance(this_output, np.ndarray):
            if CONVERTER_FUNC_NAME in mod_locals:
                converter_func = mod_locals[CONVERTER_FUNC_NAME]
                this_output = converter_func(this_output)
                assert isinstance(this_output, np.ndarray), f"{CONVERTER_FUNC_NAME} function of {tool_name} did not return an ndarray"
            else:
                raise ValueError(f"output image is not a numpy array and there is no function named {CONVERTER_FUNC_NAME} in the module")
        assert this_output.shape == correct_output.shape, f"output shape {this_output.shape} does not match millipyde output shape {correct_output.shape} (input shape: {load_image_from_path(image_path).shape})"
        assert this_output.dtype == correct_output.dtype, f"output dtype ({this_output.dtype}) does not match millipyde output dtype ({correct_output.dtype}) (input dtype: {input_image.dtype})"
        assert np.allclose(this_output, correct_output, atol=0.5, rtol=0.10), f"output image does not match millipyde output image"

    # add the test to the modules locals
    mod_locals[benchmark_name] = benchmark_func
    mod_locals[verify_name] = test_ouput_correct
