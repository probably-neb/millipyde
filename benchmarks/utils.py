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

CONVERTER_FUNC_NAME = "convert_output_to_uint8_ndarray"

RTOL = 0.10

COMPARISON_TYPE = np.float64


def ansi(s) -> str:
    return f"\033[95m{s}\033[0m"


def benchmarks_list():
    return [
        "transpose",
        "gauss_sigma_2",
        "grayscale_gauss_sigma_2",
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


def get_correct_image_path(image_path, func_name):
    image_path = Path(image_path)
    if not image_path.is_file():
        image_path = BENCHMARKS_DIR / "inputs" / image_path
    assert image_path.is_file(), f"image: {image_path} does not exist"
    image_path = path.join(
        CORRECT_IMAGE_OUTPUT_DIR, f"{image_path.stem}-{func_name}.npy"
    )
    return image_path


def wrap_setup(setup_func):
    def setup():
        image = setup_func()
        return (image,), {}

    return setup


def create_setup_func(image_path, image_from_ndarray):
    def setup():
        # TODO: memoize load image once here
        image = image_from_ndarray(load_image_from_path(image_path))
        # this wierd return signature is required by benchmark.pedantic
        # it represents *args, **kwargs
        return image

    return wrap_setup(setup)


def identity(x):
    return x


def run_benchmark(benchmark, func, image_path, rounds, image_from_ndarray=identity):
    benchmark.extra_info["image_path"] = image_path
    setup = create_setup_func(image_path, image_from_ndarray)
    benchmark.extra_info["image_path"] = image_path
    # TODO: run file once here to get output image type?
    benchmark.pedantic(func, setup=setup, rounds=rounds)


def _create_benchmark_func(func, image_from_ndarray):
    # the parameters are fixtures, i.e. keywords that tell the testing framework
    # what to pass to the function
    def benchmark_func(benchmark, image_path, rounds):
        run_benchmark(benchmark, func, image_path, rounds, image_from_ndarray)

    return benchmark_func


def create_benchmark(func, locals, image_from_ndarray=identity):
    tool_name = locals["__name__"].replace("benchmark_", "")
    benchmark_name = f"benchmark_{tool_name}_{func.__name__}"
    if not benchmark_name in locals:
        benchmark_func = _create_benchmark_func(func, image_from_ndarray)
        locals[benchmark_name] = benchmark_func


def get_millipyde_output(image_path, func_name):
    correct_output_path = get_correct_image_path(image_path, func_name)
    correct_output = np.load(correct_output_path)
    return correct_output


def verify_output(actual_output, millipyde_output):
    assert (
        actual_output.dtype == COMPARISON_TYPE
    ), f"""output dtype ({actual_output.dtype}) does not match {COMPARISON_TYPE} which is the dtype used for comparison.
Use the {convert_image_type_to_float.__name__} function in utils to convert the image to the correct type"""
    assert (
        actual_output.shape == millipyde_output.shape
    ), f"output shape {actual_output.shape} does not match millipyde output shape {millipyde_output.shape}"
    try:
        np.testing.assert_allclose(actual_output, millipyde_output, rtol=RTOL)
    except AssertionError as e:
        msg = e.args[0]
    assert np.allclose(
        actual_output, millipyde_output, rtol=RTOL
    ), f"output image does not match millipyde output image {msg}"


def convert_image_type_to_float(image):
    assert isinstance(image,np.ndarray), f"convert_image_type_to_float: expected image to be of type: np.ndarray but found: {type(image)}"
    from skimage.util import img_as_float64

    return img_as_float64(image)


def create_output_verifier(
    func,
    mod_locals,
    image_from_ndarray=identity,
    image_to_ndarray=np.asarray,
):
    tool_name = mod_locals["__name__"].replace("benchmark_", "")

    def test_ouput_correct(image_path):
        import skimage

        input_image = image_from_ndarray(load_image_from_path(image_path))
        output = image_to_ndarray(func(input_image))
        allowed_types = [np.uint8, np.float64]
        assert output.dtype in allowed_types, f"expected dtype of resulting np.array to be np.uint8 or np.float64 but found {output.dtype}"
        output = convert_image_type_to_float(output)

        millipyde_output = get_millipyde_output(image_path, func.__name__)

        verify_output(output, millipyde_output)

    verify_name = f"test_{tool_name}_{func.__name__}_output"
    if not verify_name in mod_locals:
        mod_locals[verify_name] = test_ouput_correct


def load_funcs(
    mod_locals,
    image_from_ndarray=identity,
    image_to_ndarray=np.asarray,
):
    """loads functions from a modules locals
    wraps them in what pytest-benchmark wants for a benchmark
    and puts them back"""

    for func_name in benchmarks_list():
        if func_name in mod_locals:
            mod_func = mod_locals[func_name]
            create_benchmark(mod_func, mod_locals, image_from_ndarray)
            create_output_verifier(
                mod_func, mod_locals, image_from_ndarray, image_to_ndarray
            )

            # TODO:
            # def benchmark_load_image_time
