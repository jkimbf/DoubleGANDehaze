import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import cv2 
from PIL import Image

import torchvision


# from https://kornia.readthedocs.io/en/latest/_modules/kornia/color/ycbcr.html
def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)


def get_laplacian_kernel2d(kernel_size: int) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.
    Args:
        kernel_size: filter size should be odd.
    Returns:
        2D tensor with laplacian filter matrix coefficients.
    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`
    Examples:
        >>> get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])
        >>> get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
        raise TypeError(f"ksize must be an odd positive integer. Got {kernel_size}")

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d


# img = cv2.imread('06_GT.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
# img = torchvision.transforms.functional.to_tensor(img).float()
# img = img[None, :, :, :]
# ## Assume we're given the minibatch of image tensor in yCrCb Channel

# ## TODO 1 - convert img into gray scale
# # register functional in __init__ method 
# # (refer to https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689/6?u=ptrblck)
# img = ycbcr_to_rgb(img) # img in rgb scale
# print(img.shape)
# img = torchvision.transforms.Grayscale(num_output_channels=1)(img)
# print(img.shape)

# ## TODO 2 - applay Laplacian operator on image
# print(get_laplacian_kernel2d(17))
# kernel = torch.unsqueeze(get_laplacian_kernel2d(3), dim=0)
# kernel = kernel[None, :, :, :]
# print(img.shape, kernel.shape)
# img = F.conv2d(img, kernel, groups=1, padding=3//1, stride = 1)

# # Test - convert from tensor to PIL image
# img = torchvision.transforms.ToPILImage()(img.squeeze())
# img.show()