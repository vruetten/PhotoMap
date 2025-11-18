"""
Plotting and Results Visualization Module

This module provides functions for visualizing photomap grid results,
including:
- Grid quality metrics (gridiness) computation
- Visualization of grid points overlaid on images
- Distribution plots for grid deviation metrics

Example usage:
    import numpy as np
    import tifffile as tf
    import matplotlib.pyplot as pl
    from photomap import plot_results
    
    # Load grid results
    grid_results = np.load('path/to/grid_results.npy', allow_pickle=True).item()
    grid_points = grid_results['grid_points']
    
    # Compute grid quality
    gridiness = plot_results.compute_gridiness(grid_points[:, :, 1:], L=90)
    
    # Visualize
    plot_results.plot_grid_on_image(image, grid_points, plane_idx=0)
"""

import numpy as np
from glob import glob
import tifffile as tf
import matplotlib.pyplot as pl
import matplotlib as mpl
from matplotlib import colors as mcolors
import skimage
import warnings
from pathlib import Path
from scipy.stats import gaussian_kde

# ============================================================================
# GRID QUALITY METRICS
# ============================================================================

def compute_gridiness(grid_yx, L=90, weights=(1.0, 1.0, 0.5)):
    """
    Compute grid quality metrics for detected grid points.
    
    This function quantifies how well detected points form a regular grid
    by measuring three types of deviations:
    1. Length deviation: how much edge lengths differ from expected spacing L
    2. Orthogonality deviation: how much angles differ from 90 degrees
    3. Symmetry deviation: how much opposite edges differ
    
    Parameters
    ----------
    grid_yx : ndarray, shape (ny, nx, 2) or (ny, nx, 3)
        Grid of detected points with (y, x) or (z, y, x) coordinates.
        NaN values indicate missing points.
    L : float, optional
        Expected grid spacing in pixels (default: 90).
    weights : tuple of 3 floats, optional
        Relative weights for (length, orthogonality, symmetry) deviations
        (default: (1.0, 1.0, 0.5)).
        
    Returns
    -------
    G : ndarray, shape (ny, nx, 3)
        Grid quality metrics for each point:
        G[:, :, 0] = length deviation (0=perfect, higher=worse)
        G[:, :, 1] = orthogonality deviation (0=perfect, higher=worse)
        G[:, :, 2] = symmetry deviation (0=perfect, higher=worse)
    """
    ny, nx, _ = grid_yx.shape
    G = np.full((ny, nx, 3), np.nan)
    
    w1, w2, w3 = weights
    
    for i in range(ny):
        for j in range(nx):
            center = grid_yx[i, j]
            if np.any(np.isnan(center)):
                continue
            
            # Find valid neighbors in 4 directions
            neighbors = {}
            for dy, dx, name in [(-1, 0, 'up'), (1, 0, 'down'), 
                                  (0, -1, 'left'), (0, 1, 'right')]:
                ni, nj = i + dy, j + dx
                if 0 <= ni < ny and 0 <= nj < nx:
                    neighbor = grid_yx[ni, nj]
                    if not np.any(np.isnan(neighbor)):
                        neighbors[name] = neighbor
            
            # Compute length deviation
            deviations = []
            for direction in ['right', 'down']:
                if direction in neighbors:
                    v = neighbors[direction] - center
                    deviations.append(1 - abs(np.linalg.norm(v) / L))
            length_dev = np.mean(deviations) if deviations else np.nan
            
            # Compute orthogonality deviation
            if 'right' in neighbors and 'down' in neighbors:
                vx = neighbors['right'] - center
                vy = neighbors['down'] - center
                vx_norm = np.linalg.norm(vx)
                vy_norm = np.linalg.norm(vy)
                if vx_norm > 0 and vy_norm > 0:
                    cos_angle = np.dot(vx, vy) / (vx_norm * vy_norm)
                    ortho_dev = cos_angle  # 0 = perpendicular, ±1 = parallel
                else:
                    ortho_dev = np.nan
            else:
                ortho_dev = np.nan
            
            # Compute symmetry deviation
            sym_devs = []
            if 'left' in neighbors and 'right' in neighbors:
                v1 = neighbors['left'] - center
                v2 = neighbors['right'] - center
                sym_devs.append(np.linalg.norm(v1 + v2) / L)
            if 'up' in neighbors and 'down' in neighbors:
                v1 = neighbors['up'] - center
                v2 = neighbors['down'] - center
                sym_devs.append(np.linalg.norm(v1 + v2) / L)
            sym_dev = np.mean(sym_devs) if sym_devs else 0
            
            # Store individual metrics
            G[i, j, 0] = length_dev
            G[i, j, 1] = ortho_dev
            G[i, j, 2] = sym_dev
    
    return G


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_grid_on_image(image, grid_points, plane_idx=0, metric=None, 
                        cmap='magma_r', vmin=0, vmax=0.5, figsize=(8, 16)):
    """
    Visualize grid points overlaid on an image with optional quality metrics.
    
    Parameters
    ----------
    image : ndarray, shape (Y, X)
        2D image to display as background.
    grid_points : ndarray, shape (num_planes, num_rows, num_columns, 3)
        Grid of detected points.
    plane_idx : int, optional
        Which plane to visualize (default: 0).
    metric : ndarray, shape (num_rows, num_columns), optional
        Grid quality metric to color points (default: None).
        If None, points are colored by row index.
    cmap : str, optional
        Colormap name for metric visualization (default: 'magma_r').
    vmin, vmax : float, optional
        Color scale limits for metric (default: 0, 0.5).
    figsize : tuple, optional
        Figure size in inches (default: (8, 16)).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    fig, ax = pl.subplots(figsize=figsize)
    
    # Display image
    ax.imshow(image, cmap='gray')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Extract points for this plane
    subpoints = grid_points[plane_idx, :, :, 1:]  # (rows, cols, 2) with (y, x)
    num_rows, num_columns = subpoints.shape[:2]
    
    # Plot points
    if metric is not None:
        # Color by metric
        points_flat = subpoints.reshape(-1, 2)
        metric_flat = metric.reshape(-1)
        vnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)
        sc = ax.scatter(points_flat[:, 1], points_flat[:, 0], 
                       c=metric_flat, s=20, norm=vnorm, cmap=cmap)
        pl.colorbar(sc, ax=ax, label='Grid Metric', shrink=0.2)
    else:
        # Color by row
        for row in range(num_rows):
            color = pl.get_cmap('prism')(row / num_rows)
            ax.plot(subpoints[row, :, 1], subpoints[row, :, 0], 
                   '.-', alpha=1, color=color)
    
    # Draw grid lines
    for row in range(num_rows):
        ax.plot(subpoints[row, :, 1], subpoints[row, :, 0], 
               '-', alpha=0.3, color='k', zorder=-1, lw=1)
    for col in range(num_columns):
        ax.plot(subpoints[:, col, 1], subpoints[:, col, 0], 
               '-', alpha=0.3, color='k', zorder=-1, lw=1)
    
    pl.tight_layout()
    return fig, ax


def plot_deviation_distributions(gridiness, grid_labels, on_sample_label=1,
                                 save_path=None):
    """
    Plot distributions of grid deviation metrics comparing on-sample vs off-sample.
    
    Parameters
    ----------
    gridiness : ndarray, shape (..., 3)
        Grid quality metrics from compute_gridiness.
    grid_labels : ndarray, same shape as gridiness[..., 0]
        Labels indicating which points are on sample vs off.
    on_sample_label : int, optional
        Label value for on-sample points (default: 1).
    save_path : str, optional
        If provided, save figures to this directory (default: None).
        
    Returns
    -------
    None
        Creates matplotlib figures showing KDE plots of deviations.
    """
    length_dev = gridiness[..., 0]
    ortho_dev = gridiness[..., 1]
    sym_dev = gridiness[..., 2]
    
    # Separate on-sample and off-sample
    length_on = length_dev[grid_labels == on_sample_label]
    length_off = length_dev[grid_labels != on_sample_label]
    ortho_on = ortho_dev[grid_labels == on_sample_label]
    ortho_off = ortho_dev[grid_labels != on_sample_label]
    
    # Plot length deviation
    fig, ax = pl.subplots(figsize=(2, 1.5))
    ax.set_title("Length Deviation")
    _plot_kde_comparison(length_on, length_off, ax, 
                        labels=['on sample', 'off sample'])
    ax.set_xlabel('Length deviation')
    ax.set_ylabel('Density')
    if save_path:
        fig.savefig(f'{save_path}/length_deviation_kde.pdf')
    
    # Plot angle deviation
    fig, ax = pl.subplots(figsize=(2, 1.5))
    ax.set_title("Angle Deviation")
    _plot_kde_comparison(ortho_on, ortho_off, ax,
                        labels=['on sample', 'off sample'])
    ax.set_xlabel('Angle deviation')
    ax.set_ylabel('Density')
    if save_path:
        fig.savefig(f'{save_path}/angle_deviation_kde.pdf')
    
    pl.show()


def _plot_kde_comparison(data0, data1, ax, labels=None, xlim=(-0.5, 0.5)):
    """Helper function to plot KDE comparison between two datasets."""
    # Remove NaN values
    data0 = data0[~np.isnan(data0)]
    data1 = data1[~np.isnan(data1)]
    
    if len(data0) == 0 or len(data1) == 0:
        print("Warning: Empty data for KDE plot")
        return
    
    # Compute KDEs
    kde0 = gaussian_kde(data0)
    kde1 = gaussian_kde(data1)
    
    x_eval = np.linspace(xlim[0], xlim[1], 500)
    pdf0 = kde0(x_eval)
    pdf1 = kde1(x_eval)
    
    # Plot
    if labels is None:
        labels = ['Group 0', 'Group 1']
    
    ax.plot(x_eval, pdf1, color=pl.get_cmap('tab10')(0), lw=2, alpha=1, 
           label=labels[1], zorder=-1)
    ax.fill_between(x_eval, pdf1, 0, color=pl.get_cmap('tab10')(0), 
                   alpha=0.8, zorder=-1)
    
    ax.plot(x_eval, pdf0, color=pl.get_cmap('tab10')(1), lw=2, alpha=0.7,
           label=labels[0])
    ax.fill_between(x_eval, pdf0, 0, color=pl.get_cmap('tab10')(1), 
                   alpha=0.7)
    
    ax.set_xlim(xlim)
    ax.legend(loc='upper right', fontsize='small')
