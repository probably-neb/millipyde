import pytest

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


def load_image_from_path(path: str = "./inputs/charlie1.png"):
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

    test_name = "test_" + tool_name

    # add the test to the modules locals
    mod_locals[test_name] = benchmark_func
