"""
Torch helpers shared by GPU-rendered videos: device choice, Gaussian blur/bloom
(`wide_blur` for large radii) and bilinear point splatting. Images are (H, W, C) float tensors.
"""

import torch
import torch.nn.functional as F
from loguru import logger


def select_device() -> torch.device:
    """CUDA when available, else CPU; logs which."""
    if torch.cuda.is_available():
        logger.info(f"rendering on {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.info("no CUDA device, rendering on CPU")
    return torch.device("cpu")


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    """Normalised 1-D Gaussian of radius 3 sigma, for `blur`."""
    radius = max(1, int(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    return kernel / kernel.sum()


def blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur of an (H, W, C) image, edges replicated."""
    planes = image.permute(2, 0, 1)[None]
    channels = planes.shape[1]
    radius = (kernel.numel() - 1) // 2
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    out = F.conv2d(
        F.pad(planes, (radius, radius, 0, 0), mode="replicate"),
        horizontal,
        groups=channels,
    )
    out = F.conv2d(
        F.pad(out, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels
    )
    return out[0].permute(1, 2, 0)


def wide_blur(image: torch.Tensor, sigma: float, factor: int) -> torch.Tensor:
    """
    Gaussian blur of an (H, W, C) image with a large `sigma` (in full-size pixels),
    computed on a copy shrunk `factor` times and scaled back up: wide bloom at a
    fraction of the cost of `blur`.
    """
    planes = F.avg_pool2d(image.permute(2, 0, 1)[None], factor)
    small = blur(
        planes[0].permute(1, 2, 0), gaussian_kernel(sigma / factor, image.device)
    )
    out = F.interpolate(
        small.permute(2, 0, 1)[None],
        size=image.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    return out[0].permute(1, 2, 0)


def splat(
    canvas: torch.Tensor, px: torch.Tensor, py: torch.Tensor, colours: torch.Tensor
):
    """
    Add points at pixel coordinates (px, py) into a contiguous (H, W, C) canvas,
    in place, each spread bilinearly over its four nearest pixels. `colours` is
    (N, C); points outside the canvas are dropped.
    """
    height, width, channels = canvas.shape
    flat = canvas.view(-1, channels)
    px, py = px.reshape(-1), py.reshape(-1)
    x0 = torch.floor(px)
    y0 = torch.floor(py)
    fx = px - x0
    fy = py - y0
    x0 = x0.to(torch.long)
    y0 = y0.to(torch.long)
    for ox, oy, corner in (
        (0, 0, (1.0 - fx) * (1.0 - fy)),
        (1, 0, fx * (1.0 - fy)),
        (0, 1, (1.0 - fx) * fy),
        (1, 1, fx * fy),
    ):
        ix = x0 + ox
        iy = y0 + oy
        inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        index = iy.clamp(0, height - 1) * width + ix.clamp(0, width - 1)
        flat.index_add_(0, index, colours * (corner * inside)[:, None])
