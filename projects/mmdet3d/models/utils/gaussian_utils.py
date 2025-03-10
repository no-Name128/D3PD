import torch
import numpy as np


def calculate_box_mask_gaussian(
    preds_shape, target, pc_range, voxel_size, out_size_scale
):
    B = preds_shape[0]
    C = preds_shape[1]
    H = preds_shape[2]
    W = preds_shape[3]
    gt_mask = np.zeros((B, H, W), dtype=np.float32)  # C * H * W

    for i in range(B):
        for j in range(len(target[i])):
            if target[i][j].sum() == 0:
                break

            w, h = (
                target[i][j][3] / (voxel_size[0] * out_size_scale),
                target[i][j][4] / (voxel_size[1] * out_size_scale),
            )
            radius = gaussian_radius((w, h))
            radius = max(0, int(radius))

            center_heatmap = [
                int((target[i][j][0] - pc_range[0]) / (voxel_size[0] * out_size_scale)),
                int((target[i][j][1] - pc_range[1]) / (voxel_size[1] * out_size_scale)),
            ]
            draw_umich_gaussian(gt_mask[i], center_heatmap, radius)

    gt_mask_torch = torch.from_numpy(gt_mask).cuda()
    return gt_mask_torch


def gaussian_radius(bbox_size, min_overlap=0.7):
    height, width = bbox_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
