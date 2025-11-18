"""
Grid Finding Module

This module provides functions and workflows for extracting and organizing
grid points from photomap images. It includes functionality for:
- Template matching and blob detection
- Grid point extraction and clustering
- Point deduplication and organization into structured grids
- Flattening 3D image stacks based on detected grid planes

Example usage:
    import tifffile as tf
    import numpy as np
    from photomap import find_grid, utils
    
    # Load your 3D image data
    data = tf.imread('path/to/image.tif')
    templates = np.load('path/to/templates.npy', allow_pickle=True).item()
    
    # Extract grid points from each plane
    # (See example workflows in documentation)
"""

import numpy as np
from glob import glob
import tifffile as tf
import matplotlib.pyplot as pl
import matplotlib as mpl
import skimage
import os
import warnings
from pathlib import Path
from scipy.ndimage import maximum_filter, binary_dilation, binary_erosion
from scipy.spatial import KDTree
from time import time

# Note: This file contains utility functions for grid finding workflows.
# Users should adapt paths and parameters to their specific datasets.

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def remap_labels(labels, points, axis=1):
    """
    Remap cluster labels to be ordered by median coordinate along a specified axis.
    
    This function sorts clusters based on their median position along a given axis
    and reassigns labels to reflect this spatial ordering.
    
    Parameters
    ----------
    labels : ndarray, shape (N,)
        Original cluster labels for each point.
    points : ndarray, shape (N, D)
        Point coordinates (D dimensions).
    axis : int, optional
        Axis along which to sort clusters (default: 1).
        For 3D data: 0=z, 1=y, 2=x
        
    Returns
    -------
    new_labels : ndarray, shape (N,)
        Remapped labels ordered by median position along the specified axis.
    """
    unique_labels = np.unique(labels)
    # Sort the groups by their median value along the specified axis
    all_medians = []
    for label in unique_labels:
        indices_ = np.argwhere(labels == label)[:, 0]
        subpoints = points[indices_]
        median_val = np.median(subpoints[:, axis])
        all_medians.append(median_val)
    all_medians = np.array(all_medians)
    order = np.argsort(all_medians)
    
    # Reassign labels based on sorted order
    new_labels = np.zeros_like(labels)
    for ind, order_ind in enumerate(order):
        inds_ = np.argwhere(labels == unique_labels[order_ind])[:, 0]
        new_labels[inds_] = unique_labels[ind]
    return new_labels


def reorganize_points_into_grid(points, row_labels, column_labels, plane_labels, onsample_labels):
    """
    Reorganize labeled points into a structured 3D grid array.
    
    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Point coordinates in (z, y, x) format.
    row_labels : ndarray, shape (N,)
        Row index for each point.
    column_labels : ndarray, shape (N,)
        Column index for each point.
    plane_labels : ndarray, shape (N,)
        Plane (z-slice) index for each point.
    onsample_labels : ndarray, shape (N,)
        Sample/specimen labels for each point.
        
    Returns
    -------
    grid_results : dict
        Dictionary containing:
        - 'grid_points': (num_planes, num_rows, num_columns, 3) grid array
        - 'grid_labels': (num_planes, num_rows, num_columns) sample labels
        - 'grid_row_labels': grid of row indices
        - 'grid_column_labels': grid of column indices
        - 'napari_points': flattened points for visualization
        - 'napari_row_labels', 'napari_column_labels', 'napari_fish_labels': 
          flattened label arrays
    """
    num_rows = len(np.unique(row_labels))
    num_columns = len(np.unique(column_labels))
    num_planes = len(np.unique(plane_labels))
    
    # Initialize grid arrays with NaN for missing points
    grid_points = np.nan * np.ones((num_planes, num_rows, num_columns, 3))
    grid_labels = np.zeros((num_planes, num_rows, num_columns))
    grid_row_labels = np.zeros((num_planes, num_rows, num_columns))
    grid_column_labels = np.zeros((num_planes, num_rows, num_columns))
    
    # Fill in the grid with actual point data
    grid_points[plane_labels, row_labels, column_labels] = points
    grid_labels[plane_labels, row_labels, column_labels] = onsample_labels
    grid_row_labels[plane_labels, row_labels, column_labels] = row_labels
    grid_column_labels[plane_labels, row_labels, column_labels] = column_labels
    
    # Prepare flattened arrays for visualization (e.g., Napari)
    napari_points = grid_points.copy()
    napari_points[:, :, :, 0] = np.arange(grid_points.shape[0])[:, None, None]
    napari_points = napari_points.reshape(-1, 3)
    napari_row_labels = grid_row_labels.reshape(-1)
    napari_column_labels = grid_column_labels.reshape(-1)
    napari_fish_labels = grid_labels.reshape(-1)
    
    # Package results
    grid_results = {
        'grid_points': grid_points,
        'grid_labels': grid_labels,
        'grid_row_labels': grid_row_labels,
        'grid_column_labels': grid_column_labels,
        'napari_points': napari_points,
        'napari_row_labels': napari_row_labels,
        'napari_column_labels': napari_column_labels,
        'napari_fish_labels': napari_fish_labels
    }
    
    return grid_results


def flatten_image_by_grid_planes(data, grid_points, delta_xy=250, delta_z=0):
    """
    Flatten 3D image stack into 2D planes based on detected grid points.
    
    This function creates flat 2D projections by extracting regions around
    detected grid points in each plane, then taking the minimum projection
    across the z-axis within masked regions.
    
    Parameters
    ----------
    data : ndarray, shape (Z, Y, X)
        3D image stack to flatten.
    grid_points : ndarray, shape (num_planes, num_rows, num_columns, 3)
        Grid of detected points with (z, y, x) coordinates.
        NaN values indicate missing points.
    delta_xy : int, optional
        Spatial extent around each point in x-y dimensions (default: 250).
    delta_z : int, optional
        Spatial extent around each point in z dimension (default: 0).
        
    Returns
    -------
    planes : ndarray, shape (num_planes, Y, X)
        Flattened 2D planes, one per grid plane.
    """
    num_planes = grid_points.shape[0]
    planes = []
    
    for plane_ind in range(num_planes):
        # Create binary mask for this plane's grid points
        dummy_vol = np.zeros_like(data)
        
        pointsz_ = grid_points[plane_ind, :, :, 0]
        pointsy_ = grid_points[plane_ind, :, :, 1]
        pointsx_ = grid_points[plane_ind, :, :, 2]
        
        # Flatten and filter out nan points
        valid_mask = ~np.isnan(pointsz_) & ~np.isnan(pointsy_) & ~np.isnan(pointsx_)
        zs = pointsz_[valid_mask].astype(int)
        ys = pointsy_[valid_mask].astype(int)
        xs = pointsx_[valid_mask].astype(int)
        
        # Mark regions around each valid point
        for z, y, x in zip(zs, ys, xs):
            zmin, zmax = max(z - delta_z, 0), min(z + delta_z + 1, dummy_vol.shape[0])
            ymin, ymax = max(y - delta_xy, 0), min(y + delta_xy + 1, dummy_vol.shape[1])
            xmin, xmax = max(x - delta_xy, 0), min(x + delta_xy + 1, dummy_vol.shape[2])
            dummy_vol[zmin:zmax, ymin:ymax, xmin:xmax] = 1
        
        # Find the z-plane with most coverage
        plane_with_most_points = np.argmax(dummy_vol.sum(axis=(1, 2)))
        
        # Ensure full x-y coverage (fill gaps from best plane)
        coverage_mask = dummy_vol.sum(axis=0)  # shape: (Y, X)
        if np.any(coverage_mask == 0):
            dummy_vol[plane_with_most_points, coverage_mask == 0] = 1
        
        # Apply mask and project
        vol = data.copy()
        vol[dummy_vol == 0] = np.nan
        vol = np.nanmin(vol, axis=0)
        planes.append(vol)
    
    planes = np.stack(planes, axis=0)
    return planes
