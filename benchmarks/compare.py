import benchmark_cupy as bcp
import benchmark_millipyde as bmp
import benchmark_scikit_image as bsi
import benchmark_pillow as bpl
import utils
import numpy as np

a = utils.load_image_from_path("inputs/charlie12.png")

ca = bcp.cupy_array_from_ndarray(a)
rca = bcp.rotate_90_deg(ca)
cres = bcp.cupy_array_to_ndarray(rca)

pa = bpl.pil_image_from_ndarray(a)
rpa = bpl.rotate_90_deg(pa)
pres = bpl.to_ndarray(rpa)

sires = bsi.rotate_90_deg(a)

ma = bmp.gpuimage_from_ndarray(a)
rma = bmp.rotate_90_deg(ma)
mres = np.asarray(rma)

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
# plt.imshow(mres)
# plt.show()
fig = plt.figure(figsize=(8, 9))

from skimage import data, transform, exposure
from skimage.util import compare_images
def show_diff(a,aname,b,bname):
    blend_rotated = compare_images(a, b, method='diff')

    gs = GridSpec(3, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1:, :])

    ax0.imshow(a, cmap='gray')
    ax0.set_title(aname)
    ax1.imshow(b, cmap='gray')
    ax1.set_title(bname)
    ax2.imshow(blend_rotated, cmap='gray')
    ax2.set_title('Blend comparison')
    for a in (ax0, ax1, ax2):
        a.axis('off')
    plt.tight_layout()
    plt.plot()
    plt.show()
# show_diff(mres,"millipyde",cres,"cupy")
# show_diff(pres,"pillow",cres,"cupy")
show_diff(sires,"scikit-image",cres,"cupy")
zero = np.zeros_like(mres)
pc_m = (mres > 0).sum() / zero.size
pc_c = (sires > 0).sum() / zero.size
