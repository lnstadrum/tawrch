import torch
import _tawrch


class Augment(_tawrch.Augment):
    """Applies a set of random geometry and color transformations to batches of images.
    The applied transformation differs from one image to another one in the same batch.
    Transformations parameters are sampled from uniform distributions of given ranges.
    Their default values enable some moderate amount of augmentations.
    Every image is sampled only once through a bilinear interpolator.

    Transformation application order:
        - horizontal and/or vertical flipping,
        - perspective distortion,
        - in-plane image rotation and scaling,
        - translation,
        - gamma correction,
        - hue, saturation and brightness correction,
        - color inversion,
        - mixup,
        - CutOut.
    """
    pass


class CenterCrop(_tawrch.Augment):
    """Helper class to sample a center crop of maximum size from a given image.
    This can be useful when constructing batches from a set of images of different
    sizes trying to maximize the sampled area.
    The output is sampled by taking the maximum area fitting into the input image
    bounds, but keeping a given output aspect ratio.
    The output size in pixels is fixed and does not depend on the input image size.
    """

    def __init__(self, size, translation=0):
        """Creates a CenterCrop instance
        By default, i.e. with `translation=0`, the output image center matches the
        input image center. Otherwise, the output image sampling area is randomly
        shifted according to the given `translation` value and may fall out of the
        input image area. The corresponding output pixels are filled with gray.
        Bilinear interpolation is used to compute the output pixels values.

        Args:
            x:              Input image tensor  of `uint8` type in channels-last (HWC)
                            layout. The color input images are supported only (C=3).
            size:           A list or tuple `(W, H)` specifying the output image size
                            in pixels.
            translation:    Normalized image translation range along X and Y axis.
                            `0.1` corresponds to a random shift by at most 10% of the
                            image size in both directions.
        """

        super().__init__(
            seed=0,
            translation=translation,
            scale=0,
            prescale=1,
            rotation=0,
            perspective=0,
            flip_horizontally=False,
            flip_vertically=False,
            cutout=0,
            cutout_size=0,
            mixup=0,
            saturation=0,
            brightness=0,
            hue=0,
            gamma_corr=0,
            color_inversion=False,
        )
        self.size = size

    def __call__(self, x):
        return super().__call__(x, output_size=self.size)
