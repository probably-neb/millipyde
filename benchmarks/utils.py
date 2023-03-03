def ansi(s) -> str:
    return '\033[95m' + s + '\033[0m'

def benchmarks_list():
    return ["transpose", "gauss_sigma_2", "rotate_90_deg", "rgb_to_grayscale", "adjust_gamma_2_gain_1", "fliplr"]

def load_image_from_path(path: str):
    import imageio.v2 as io

    # returns a np.array
    img = io.imread(path)

    return img
