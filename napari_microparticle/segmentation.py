import numpy as np

from skimage import filters, feature, morphology
from skimage.segmentation import watershed

from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, label


edge_filter_by_name: dict = {
    'Sobel': filters.sobel,
    'Scharr': filters.scharr,
    'Farid': filters.farid,
    'Prewitt': filters.prewitt
}


def watershed_pores(img_data, edge_filter: str = 'Sobel', min_depth: int = 20, min_area: int = 10, max_area: int = 200):
    minima = morphology.h_minima(img_data, min_depth)
    markers = morphology.label(minima)

    gradient = edge_filter_by_name[edge_filter](img_data)

    labels = watershed(gradient, markers)

    pore_mask = np.zeros_like(img_data, dtype=bool)
    for label in np.unique(labels):
        if label == 0:  # background
            continue
        mask = labels == label
        area = np.sum(mask)
        if (area < min_area) or (area > max_area):
            continue

        pore_mask[mask] = True

    return pore_mask, markers


def get_particle_border_zone(particle_mask, zone_width: int = 3, background_border_only: bool = False):
    if zone_width == 0:
        return np.zeros(particle_mask.shape, dtype=bool)

    if background_border_only:
        border_mask = (particle_mask == 0)
    else:
        border_mask = np.zeros(particle_mask.shape, dtype=bool)

        # Horizontal differences (left-right)
        border_mask[:, 1:] |= (particle_mask[:, 1:] != particle_mask[:, :-1])
        border_mask[:, :-1] |= (particle_mask[:, 1:] != particle_mask[:, :-1])

        # Vertical differences (up-down)
        border_mask[1:, :] |= (particle_mask[1:, :] != particle_mask[:-1, :])
        border_mask[:-1, :] |= (particle_mask[1:, :] != particle_mask[:-1, :])

    # Create a disk structuring element of radius 3 (7x7)
    y, x = np.ogrid[-zone_width:zone_width+1, -zone_width:zone_width+1]
    disk = x * x + y * y <= zone_width**2
    return binary_dilation(border_mask, structure=disk)
