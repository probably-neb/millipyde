import utils
import numpy as np
import argparse

DEFAULT_FUNC="gauss_sigma_2"
DEFAULT_COMPARISON_IMAGE = "inputs/charlie12.png"
DEFAULT_METHOD = "checkerboard"
parser = argparse.ArgumentParser()
parser.add_argument("-f", type=str, dest="func",help="the function to compare", default=DEFAULT_FUNC)
parser.add_argument("-i", type=str, dest="image",help="the image to compare", default=DEFAULT_COMPARISON_IMAGE)
parser.add_argument("-m", type=str, dest="method", help="the comparison method", default=DEFAULT_METHOD) # allowed_values=["checkerboard","diff", "blend"]

args = parser.parse_args()
FUNC=args.func
COMPARISON_IMAGE = args.image
METHOD = args.method

a = utils.load_image_from_path(COMPARISON_IMAGE)

def cupy():
    import benchmark_cupy as bcp
    ca = bcp.cupy_array_from_ndarray(a)
    rca = getattr(bcp,FUNC)(ca)
    cres = bcp.cupy_array_to_ndarray(rca)
    return cres

def pillow():
    import benchmark_pillow as bpl
    pa = bpl.pil_image_from_ndarray(a)
    rpa = getattr(bpl,FUNC)(pa)
    pres = bpl.to_ndarray(rpa)
    return pres

def opencv():
    import benchmark_opencv as bcv
    return getattr(bcv,FUNC)(a)

def scikit_image():
    import benchmark_scikit_image as bsi
    sires = getattr(bsi,FUNC)(a)
    return sires

def run_millipyde():
    import benchmark_millipyde as bmp
    ma = bmp.gpuimage_from_ndarray(a)
    rma = getattr(bmp,FUNC)(ma)
    mres = np.asarray(rma)
    return mres

def load_millipyde():
    correct_path = utils.get_correct_image_path(COMPARISON_IMAGE, FUNC)
    mpres = np.load(correct_path)
    return mpres

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
# plt.imshow(mres)
# plt.show()
fig = plt.figure(figsize=(8, 9))

from skimage import data, transform, exposure
from skimage.util import compare_images

def show_diff(a,aname,b,bname):
    blend_rotated = compare_images(a, b, method=METHOD, n_tiles=(16,16))

    gs = GridSpec(3, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1:, :])

    ax0.imshow(a, cmap='gray')
    ax0.set_title(aname)
    ax1.imshow(b, cmap='gray')
    ax1.set_title(bname)
    ax2.imshow(blend_rotated, cmap='gray')
    ax2.set_title(METHOD + ' comparison')
    for x in (ax0, ax1, ax2):
        x.axis('off')
    plt.tight_layout()
    plt.plot()
    a,b = map(utils.convert_image_type_to_float, (a,b))
    percent_diff = utils.percent_mismatched(a,b)
    avg_diff = np.mean(a-b)
    stdev = np.std(a-b)
    print(f"{aname} vs {bname} percent diff: {percent_diff} mean: {avg_diff} stdev: {stdev} {a.dtype} {b.dtype}")
    plt.savefig(f"./outputs/{METHOD}-comparison-{aname}-{bname}-{FUNC}.png")
    plt.show()
show_diff(load_millipyde(),"millipyde",scikit_image(),"scikit-image")
# show_diff(pres,"pillow",cres,"cupy")
# show_diff(scikit_image(),"scikit-image",opencv(),"opencv")

