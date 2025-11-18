"""
Plotting and Results Visualization Module

This module provides functions for visualizing photomap grid results,
including:
- Grid quality metrics (gridiness) computation
- Visualization of grid points overlaid on images
- Distribution plots for grid deviation metrics

Example usage:
    import numpy as np
    import matplotlib.pyplot as pl
    from photomap import plot_results
    
    # Compute grid quality
    gridiness = plot_results.compute_gridiness(grid_points[:, :, 1:], L=90)
    
    # Plot deviation distributions
    plot_results.plot_deviation_distributions(gridiness, grid_labels)
    
    # Visualize grid on image
    plot_results.plot_grid_with_deviation(image, grid_points[6], gridiness[6], 
                                          deviation_type='length')
"""

import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
from matplotlib import colors as mcolors
import skimage
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
# DEVIATION DISTRIBUTION PLOTS
# ============================================================================

def plot_deviation_distributions(gridiness, grid_labels, on_sample_label=1,
                                 xlim=(-0.5, 0.5), ylim=(0, 27), 
                                 figsize=(2, 1.5), save_path=None):
    """
    Plot distributions of grid deviation metrics comparing on-sample vs off-sample.
    
    This creates KDE (kernel density estimate) plots showing the distributions of
    length and angle deviations for points on the sample versus off the sample.
    
    Parameters
    ----------
    gridiness : ndarray, shape (..., 3)
        Grid quality metrics from compute_gridiness.
    grid_labels : ndarray, same shape as gridiness[..., 0]
        Labels indicating which points are on sample vs off.
    on_sample_label : int, optional
        Label value for on-sample points (default: 1).
    xlim : tuple, optional
        X-axis limits for plots (default: (-0.5, 0.5)).
    ylim : tuple, optional
        Y-axis limits for plots (default: (0, 27)).
    figsize : tuple, optional
        Figure size in inches (default: (2, 1.5)).
    save_path : str, optional
        If provided, save figures to this directory (default: None).
        
    Returns
    -------
    figs : list
        List of matplotlib figure objects [length_fig, angle_fig].
    """
    length_dev = gridiness[..., 0]
    ortho_dev = gridiness[..., 1]
    
    # Separate on-sample and off-sample
    length_on = length_dev[grid_labels == on_sample_label]
    length_off = length_dev[grid_labels != on_sample_label]
    ortho_on = ortho_dev[grid_labels == on_sample_label]
    ortho_off = ortho_dev[grid_labels != on_sample_label]
    
    figs = []
    
    # Plot length deviation
    fig, ax = pl.subplots(figsize=figsize)
    ax.set_title("Length Deviation")
    _plot_kde_comparison(length_on, length_off, ax, 
                        labels=['on sample', 'off sample'], xlim=xlim)
    ax.set_xlabel('Length deviation')
    ax.set_ylabel('Density')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if save_path:
        fig.savefig(f'{save_path}/length_deviation_kde.pdf', bbox_inches='tight')
    figs.append(fig)
    
    # Plot angle deviation
    fig, ax = pl.subplots(figsize=figsize)
    ax.set_title("Angle Deviation")
    _plot_kde_comparison(ortho_on, ortho_off, ax,
                        labels=['on sample', 'off sample'], xlim=xlim)
    ax.set_xlabel('Angle deviation')
    ax.set_ylabel('Density')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if save_path:
        fig.savefig(f'{save_path}/angle_deviation_kde.pdf', bbox_inches='tight')
    figs.append(fig)
    
    return figs


def _plot_kde_comparison(data0, data1, ax, labels=None, xlim=(-0.5, 0.5)):
    """Helper function to plot KDE comparison between two datasets."""
    # Remove NaN values
    data0 = data0.flatten()
    data1 = data1.flatten()
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
    
    # Plot off-sample first (background)
    ax.plot(x_eval, pdf1, color=pl.get_cmap('tab10')(0), lw=2, alpha=1, 
           label=labels[1], zorder=-1)
    ax.fill_between(x_eval, pdf1, 0, color=pl.get_cmap('tab10')(0), 
                   alpha=0.8, zorder=-1)
    
    # Plot on-sample on top
    ax.plot(x_eval, pdf0, color=pl.get_cmap('tab10')(1), lw=2, alpha=0.7,
           label=labels[0])
    ax.fill_between(x_eval, pdf0, 0, color=pl.get_cmap('tab10')(1), 
                   alpha=0.7)


# ============================================================================
# GRID VISUALIZATION ON IMAGES
# ============================================================================

def plot_grid_with_deviation(image, grid_points_2d, gridiness_2d, deviation_type='length',
                             vmin=0, vmax=0.5, figsize=(8, 16), save_path=None):
    """
    Visualize grid points overlaid on an image with deviation metrics as colors.
    
    Parameters
    ----------
    image : ndarray, shape (Y, X)
        2D image to display as background.
    grid_points_2d : ndarray, shape (num_rows, num_columns, 2 or 3)
        Grid of detected points for a single plane.
        If shape is (..., 3), uses last 2 dimensions as (y, x).
    gridiness_2d : ndarray, shape (num_rows, num_columns, 3)
        Grid quality metrics for the same plane.
    deviation_type : str, optional
        Type of deviation to visualize: 'length' or 'angle' (default: 'length').
    vmin, vmax : float, optional
        Color scale limits for deviation metric (default: 0, 0.5).
    figsize : tuple, optional
        Figure size in inches (default: (8, 16)).
    save_path : str, optional
        If provided, save figure to this path (default: None).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    fig, ax = pl.subplots(figsize=figsize)
    
    # Display image
    im = ax.imshow(image, cmap='gray', vmin=40, vmax=200)
    
    # Extract coordinates (handle both 2D and 3D point arrays)
    if grid_points_2d.shape[-1] == 3:
        subpoints = grid_points_2d[:, :, 1:]  # Take last 2 dimensions (y, x)
    else:
        subpoints = grid_points_2d
    
    # Select deviation metric
    deviation_idx = 0 if deviation_type == 'length' else 1
    sublabels = np.abs(gridiness_2d[:, :, deviation_idx])
    
    # Flatten for scatter plot
    points_flat = subpoints.reshape(-1, 2)
    labels_flat = sublabels.reshape(-1)
    
    # Create colormap normalization
    vcenter = (vmin + vmax) / 2
    vnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = pl.get_cmap('magma_r')
    
    # Plot points with colors
    sc = ax.scatter(points_flat[:, 1], points_flat[:, 0], 
                   c=labels_flat, s=20, norm=vnorm, cmap=cmap)
    pl.colorbar(sc, ax=ax, label=f'{deviation_type} deviation', shrink=0.2)
    
    # Draw grid lines
    num_rows, num_columns = subpoints.shape[:2]
    for row in range(num_rows):
        ax.plot(subpoints[row, :, 1], subpoints[row, :, 0], 
               '-', alpha=1, color='k', zorder=-1, lw=1)
    for col in range(num_columns):
        ax.plot(subpoints[:, col, 1], subpoints[:, col, 0], 
               '-', alpha=1, color='k', zorder=-1, lw=1)
    
    ax.axis('equal')
    ax.axis('off')
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig, ax


def plot_grid_by_rows_and_columns(image, grid_points_2d, figsize=(8, 16), 
                                  gamma=1.2, save_path=None):
    """
    Visualize grid with rows and columns colored differently.
    
    Parameters
    ----------
    image : ndarray, shape (Y, X)
        2D image to display as background.
    grid_points_2d : ndarray, shape (num_rows, num_columns, 2 or 3)
        Grid of detected points for a single plane.
        If shape is (..., 3), uses last 2 dimensions as (y, x).
    figsize : tuple, optional
        Figure size in inches (default: (8, 16)).
    gamma : float, optional
        Gamma correction for image display (default: 1.2).
    save_path : str, optional
        If provided, save figure to this path (default: None).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    fig, ax = pl.subplots(figsize=figsize)
    
    # Apply gamma correction to image
    image_adj = image - image.min()
    image_gamma = skimage.exposure.adjust_gamma(image_adj, gamma)
    
    ax.imshow(image_gamma, cmap='gray')
    
    # Extract coordinates
    if grid_points_2d.shape[-1] == 3:
        subpoints = grid_points_2d[:, :, :]
    else:
        subpoints = grid_points_2d
    
    num_rows, num_columns = subpoints.shape[:2]
    
    # Plot rows with different colors
    for ind, row in enumerate(range(num_rows)):
        color = pl.get_cmap('prism')(ind / num_rows)
        if subpoints.shape[-1] == 3:
            ax.plot(subpoints[row, :, 2], subpoints[row, :, 1], 
                   '.-', alpha=1, color=color)
        else:
            ax.plot(subpoints[row, :, 1], subpoints[row, :, 0], 
                   '.-', alpha=1, color=color)
    
    # Plot columns with different colors
    for ind, col in enumerate(range(num_columns)):
        color = pl.get_cmap('prism')(ind / num_columns)
        if subpoints.shape[-1] == 3:
            ax.plot(subpoints[:, col, 2], subpoints[:, col, 1], 
                   '.-', alpha=1, color=color)
        else:
            ax.plot(subpoints[:, col, 1], subpoints[:, col, 0], 
                   '.-', alpha=1, color=color)
    
    ax.axis('equal')
    ax.axis('off')
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    
    return fig, ax


def plot_grid_spacing_analysis(grid_points_2d, figsize=(5, 3)):
    """
    Plot grid point spacing along rows and columns.
    
    Parameters
    ----------
    grid_points_2d : ndarray, shape (num_rows, num_columns, 2 or 3)
        Grid of detected points for a single plane.
    figsize : tuple, optional
        Figure size in inches (default: (5, 3)).
        
    Returns
    -------
    figs : list
        List of matplotlib figure objects [row_spacing_fig, col_spacing_fig].
    """
    if grid_points_2d.shape[-1] == 3:
        subpoints = grid_points_2d[:, :, 1:]  # Use last 2 dims (y, x)
    else:
        subpoints = grid_points_2d
    
    num_rows, num_columns = subpoints.shape[:2]
    figs = []
    
    # Plot rows with spacing colored
    fig, ax = pl.subplots(figsize=figsize)
    for ind, row in enumerate(range(num_rows)):
        color = pl.get_cmap('magma')(ind / num_rows)
        ax.plot(subpoints[row, :, 1], subpoints[row, :, 0], 
               'o-', alpha=1, color=color)
    ax.set_title('Grid rows')
    ax.set_aspect('equal')
    figs.append(fig)
    
    # Plot columns with spacing colored
    fig, ax = pl.subplots(figsize=figsize)
    for ind, col in enumerate(range(num_columns)):
        color = pl.get_cmap('magma')(ind / num_columns)
        ax.plot(subpoints[:, col, 1], subpoints[:, col, 0], 
               'o-', alpha=1, color=color)
    ax.set_title('Grid columns')
    ax.set_aspect('equal')
    figs.append(fig)
    
    return figs
