# Benchmarks

these benchmarks were created with the intent of comparing millipyde to similar tools

## NOTES

The outputs of non-millipyde tools are as close as possible to the outputs of millipyde, however equality between the outputs was not always possible for the reasons listed below

### Common Problems

#### 1. the behavior of millipyde's gaussian blur function for inputs with 4 channels (rgba) results in an image with the alpha channel of each pixel set to max (i.e. 255 for u8 or 1.0 for float).

This is most likely a bug as it is the only image function with behavior other than that of the equivalent skimage function. This inconsistency with skimage is not shown in the comparisons ran by `run_tests.sh` as the `test_gaussian` function converts the image to grayscale before applying the gaussian blur.

Skimage, Cupy (whose gaussian function is based on the scikit implementation), and opencv do not have this behavior.

Additionally the results of skimage, cupy, and opencv match each other, but not millipyde, even when the alpha channel is ignored.

#### 2. multiple tools have strange behavior at the edges when rotating 90 degrees

pillow, cupy, skimage, and millipyde all output images with different edges at the top and bottom of the rotated image. this is best explained with an image

**ORIGINAL**

![original image](./inputs/charlie12.png)

**AFTER 90 DEGREE ROTATION**

![diff between millipyde and skimage](./outputs/diff-comparison-millipyde-scikit-image-rotate_90_deg.png)

This difference was deemed acceptable as it only resulted in ~ 8% of pixels being mismatched in the worst case (the image above) and the difference decreased the larger the input image.

### Tool Specific Problems/Notes

#### Cupy
 
Because of how Cupy (and numpy) store arrays, a transpose is simply a permutation on the axes, i.e. it does very little work. I'm including this note to help explain why cupy's transpose is so much faster than the other tools.

#### Opencv

1. The builtin rotate method was not used for the `rotate_90_deg` function as it resized the image. Instead the opencv methods for generating and applying were used

2. opencv and scikit image compute the kernel size based on sigma differently when doing gaussian blur. For the cpu implementation of opencv the kernel size was computed the same way scikit image does as scikit image is what the millipyde functions seem to be based on, however for the cuda implementation there is an (arbitrary?) restriction on the kernel dimensions. The closest possible kernel size was used.



