from __future__ import annotations

from typing import List, Tuple


def estimate_mlp_flops(layer_shapes: List[Tuple[int, int]], batch_size: int, include_backward: bool = True) -> int:
    """Rough FLOPs estimate for one forward(+backward) pass of an MLP.

    Assumptions:
      - MatMul cost ~ 2*m*n*k (multiply-adds counted as 2 FLOPs)
      - Bias add ~ n*k
      - ReLU ~ n*k (comparison)
      - Backward roughly ~ 2-3x forward; here we approximate as equal to forward for simplicity
    Returns integer FLOPs for the whole batch.
    """
    flops_fwd = 0
    for in_dim, out_dim in layer_shapes:
        # matmul: (batch, in_dim) x (in_dim, out_dim)
        flops_fwd += 2 * batch_size * in_dim * out_dim
        # bias add
        flops_fwd += batch_size * out_dim
        # relu (except last layer)
        if (in_dim, out_dim) != layer_shapes[-1]:
            flops_fwd += batch_size * out_dim
    flops = flops_fwd
    if include_backward:
        flops += flops_fwd
    return int(flops)


def estimate_conv2d_flops(batch_size: int, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], output_hw: Tuple[int, int]) -> int:
    kh, kw = kernel_size
    oh, ow = output_hw
    return int(2 * batch_size * out_channels * in_channels * kh * kw * oh * ow)


