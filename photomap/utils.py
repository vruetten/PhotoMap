"""
Photomap Utilities Module

This module provides core image processing and analysis utilities for photomap data,
including:
- FFT-based cross-correlation for template matching
- Blob detection using Laplacian of Gaussian
- Grid construction and analysis from detected points
- Connected component analysis for spatial clustering
"""

from numpy.fft import fft2, ifft2
import matplotlib.pyplot as pl
from scipy.ndimage import gaussian_laplace
from skimage.feature import peak_local_max
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

# ============================================================================
# FFT-BASED CORRELATION METHODS
# ============================================================================

# ------------------------------------------------------------
# 1. FFT-based linear cross-correlation (un-normalised)
# ------------------------------------------------------------
def xcorr2d_fft(I, T):
    """
    Cross-correlate two real 2-D arrays via FFT.
      I : N×M reference image
      T : P×Q template (P << N, Q << M)
    Returns a (N-P+1)×(M-Q+1) correlation map.
    """
    P, Q = T.shape
    N, M = I.shape
    K = 1 << int(np.ceil(np.log2(N + P - 1)))   # next power of two for height
    L = 1 << int(np.ceil(np.log2(M + Q - 1)))   # next power of two for width
    pad_I = np.zeros((K, L), dtype=I.dtype)
    pad_T = np.zeros_like(pad_I)

    pad_I[:N, :M] = I
    pad_T[:P, :Q] = T[::-1, ::-1]               # 180° rotate for correlation

    C_full = np.fft.ifft2(np.fft.fft2(pad_I) *
                          np.conj(np.fft.fft2(pad_T))).real
    return C_full[:N - P + 1, :M - Q + 1]


# ------------------------------------------------------------
# 2. Fast sliding-window sums using an integral image
# ------------------------------------------------------------
def sliding_window_sums(A, k):
    """
    Sum of every k×k patch in A (integral-image trick).
      A : N×N array
      k : window size
    Returns an (N-k+1)×(N-k+1) array of sums.
    """
    # Build 2-D prefix sum with a 1-pixel zero pad on top/left
    II = np.pad(A, ((1, 0), (1, 0)), mode='constant').cumsum(0).cumsum(1)
    tl = II[:-k, :-k]     # top-left
    tr = II[:-k, k:]      # top-right
    bl = II[k:, :-k]      # bottom-left
    br = II[k:, k:]       # bottom-right
    return br - bl - tr + tl


# ------------------------------------------------------------
# 3. Zero-mean Normalised Cross-Correlation (values ∈ [-1, 1])
# ------------------------------------------------------------
def zncc(I, T):
    """
    Zero-mean normalised cross-correlation between square images.
    Peaks at +1 for identical patches (up to DC offset), –1 if inverted.
    """
    M = T.shape[0]
    area = M * M

    # Template statistics (scalar – compute once)
    sum_T  = T.sum()
    sum_T2 = (T ** 2).sum()
    mean_T = sum_T / area
    var_T  = sum_T2 - area * mean_T ** 2           # variance * area
    std_T  = np.sqrt(var_T)

    # Sliding sums over I and I² (same size as correlation map)
    sum_I  = sliding_window_sums(I, M)
    sum_I2 = sliding_window_sums(I ** 2, M)

    # Un-normalised correlation (FFT trick)
    C = xcorr2d_fft(I, T)

    # Numerator: subtract means
    num = C - sum_I * mean_T

    # Denominator: √(var_I * var_T)
    var_I  = sum_I2 - (sum_I ** 2) / area
    denom  = np.sqrt(var_I * var_T)

    # Avoid divide-by-zero where local variance is zero
    return np.where(denom > 0, num / denom, 0.0)

def downsample_data(test_data, factor=2):
    """
    Downsample 2D data by averaging over blocks.
    
    Parameters
    ----------
    test_data : ndarray
        Input 2D array to downsample.
    factor : int, optional
        Downsampling factor (default: 2).
        
    Returns
    -------
    ndarray
        Downsampled array with shape reduced by factor in both dimensions.
    """
    test_data_ = test_data[:(test_data.shape[0]//factor)*factor, :(test_data.shape[1]//factor)*factor].reshape(test_data.shape[0]//factor, factor, test_data.shape[1]//factor, factor,)
    test_data_ = np.mean(test_data_, axis=(1,3))
    return test_data_


# ============================================================================
# PEAK ENHANCEMENT AND DETECTION
# ============================================================================

from scipy.ndimage import uniform_filter

def enhance_peaks_znorm(C, k=31, eps=1e-7):
    """
    Local Z-score normalisation:
        (C − local_mean) / local_std

    More robust when the global dynamic range drifts across the image.
    """
    mean  = uniform_filter(C,  size=k, mode="reflect")
    mean2 = uniform_filter(C**2, size=k, mode="reflect")
    var   = np.maximum(mean2 - mean**2, 0.0)
    std   = np.sqrt(var) + eps          # avoid divide-by-zero

    return (C - mean) / std

def local_top_percentile(arr, patch=50, percentile=99):
    """
    Boolean mask of local top-percentile pixels.

    Parameters
    ----------
    arr : 2-D ndarray
        Your correlation (or intensity) map.
    patch : int
        Side length of the square window (e.g. 50).
    percentile : float ∈ [0, 100]
        e.g. 99 keeps the brightest 1 % of pixels per patch.

    Returns
    -------
    mask : bool ndarray, same shape as arr
        True where arr ≥ local percentile threshold.
    """
    h, w = arr.shape
    mask = np.zeros_like(arr, dtype=bool)

    for i in range(0, h, patch):
        for j in range(0, w, patch):
            sub = arr[i:i+patch, j:j+patch]
            if sub.size == 0:
                continue
            thr = np.percentile(sub, percentile)
            mask[i:i+patch, j:j+patch] = sub >= thr
    return mask


def find_blob_centers(image,
                      radius_px: float,
                      min_distance: int = 30,
                      threshold_rel: float = 0.01):
    """
    Detect blob centres in a bright-on-dark image.

    Parameters
    ----------
    image : 2-D ndarray (float or uint)
        Input image.
    radius_px : float
        Approximate blob radius in pixels.
    min_distance : int
        Minimum separation (px) between detected centres.
        Set to >= 30 as per your prior knowledge.
    threshold_rel : float
        Relative threshold wrt max(LoG) to decide what is “bright enough”.
        Increase if you get false positives; decrease if you miss dim blobs.

    Returns
    -------
    coords : (N, 2) ndarray
        Array of (row, col) coordinates of detected centres.
    LoG : 2-D ndarray
        The Laplacian-of-Gaussian response (useful for debugging / tuning).
    """
    # 1. Scale–space parameter
    sigma = radius_px / np.sqrt(2)

    # 2–3. LoG response (SciPy’s sign convention gives negative peaks)
    LoG = -gaussian_laplace(image.astype(np.float32), sigma=sigma)

    # 4. Threshold + non-maximum suppression
    coords = peak_local_max(
        LoG,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        exclude_border=False,
        # indices=True,          # ✔ rows/cols as an (N,2) array
        num_peaks=np.inf,
        footprint=None
    )
    return coords, LoG


# ============================================================================
# GRID CLUSTERING AND CONNECTED COMPONENTS
# ============================================================================

def horizontal_link_clustering(P, k=6, max_dist=1.0, min_dist=0.1, curvature_deg=20, margin = np.pi/10, single_angle_max = 15, weighted_dist_max = 1000):
    """
    Cluster points by linking to nearest neighbors in horizontal/vertical directions.
    
    This function builds a grid-like structure by connecting each point to its
    nearest neighbor in four quadrant directions (right, up, left, down).
    
    Parameters
    ----------
    P : ndarray, shape (N, 2) or (N, 3)
        Array of 2D or 3D points to cluster.
    k : int, optional
        Number of k-nearest neighbors to consider (default: 6).
    max_dist : float, optional
        Maximum distance for valid neighbor links (default: 1.0).
    min_dist : float, optional
        Minimum distance for valid neighbor links (default: 0.1).
    curvature_deg : float, optional
        Maximum turning angle in degrees (currently unused, default: 20).
    margin : float, optional
        Angular margin for quadrant assignment in radians (default: π/10).
    single_angle_max : float, optional
        Maximum angle deviation in degrees for single neighbors (default: 15).
    weighted_dist_max : float, optional
        Maximum weighted distance threshold (default: 1000).
        
    Returns
    -------
    neighbor_idxs : ndarray, shape (N, 4)
        Indices of neighbors for each point: [right, up, left, down].
        -1 indicates no valid neighbor in that direction.
    """
    from collections import defaultdict
    
    N = len(P)
    tree = KDTree(P)
    
    # Step 1: link each point to nearest 'most horizontal' neighbour
    neighbor_idxs = []
    links = -np.ones(N, dtype=int)
    for i in range(N):
        dists, idxs = tree.query(P[i:i+1], k=k+1)      
        upper_quadrants_idx = []
        upper_quadrants_dist = []
        upper_quadrants_angle = []
        lower_quadrants_idx = []
        lower_quadrants_dist = []
        lower_quadrants_angle = []
        left_quadrants_idx = []
        left_quadrants_angle = []
        left_quadrants_dist = []
        right_quadrants_idx = []
        right_quadrants_angle = []
        right_quadrants_dist = []
        # weight the distance by the angle
        def f(x):
            x = np.abs(x/(np.pi/2))
            return np.exp(2*x)

        def wrap_angle(angle):
            theta_wrapped = (angle + np.pi) % (2*np.pi) - np.pi  
            return theta_wrapped

        for d, j in zip(dists[0,1:], idxs[0,1:]):
            if d > max_dist: # if too large ignore
                continue
            if d < min_dist: # if too small ignore
                continue
            angle = np.arctan2(P[j,2] - P[i,2], P[j,1] - P[i,1])
            if angle > -1*np.pi/4+margin and angle < 1*np.pi/4-margin:
                right_quadrants_idx.append(j)
                ideal_angle = 0
                angle_dist = wrap_angle(np.abs(angle - ideal_angle))
                weighted_dist = (f(angle_dist)+1)*d
                if weighted_dist > weighted_dist_max:
                    print(f"weighted_dist: {weighted_dist}")
                else:
                    right_quadrants_dist.append(weighted_dist)
                    right_quadrants_angle.append(angle_dist)
            elif angle > 1*np.pi/4+margin and angle < 3*np.pi/4-margin:
                upper_quadrants_idx.append(j)
                ideal_angle = np.pi/2
                angle_dist = wrap_angle(np.abs(angle - ideal_angle))
                weighted_dist = (f(angle_dist)+1)*d
                upper_quadrants_dist.append(weighted_dist)
                upper_quadrants_angle.append(angle_dist)
            elif angle > -3*np.pi/4+margin and angle < -1*np.pi/4-margin:
                lower_quadrants_idx.append(j)
                ideal_angle =  -np.pi/2
                angle_dist = wrap_angle(np.abs(angle - ideal_angle))
                weighted_dist = (f(angle_dist)+1)*d
                lower_quadrants_dist.append(weighted_dist)
                lower_quadrants_angle.append(angle_dist)
            elif (angle > 3*np.pi/4+margin and angle < 5*np.pi/4-margin) or (angle > -5*np.pi/4+margin and angle < -3*np.pi/4-margin):
                left_quadrants_idx.append(j)
                ideal_angle = -np.pi
                angle_dist = wrap_angle(np.abs(angle - ideal_angle))
                weighted_dist = (f(angle_dist)+1)*d
                left_quadrants_dist.append(weighted_dist)
                left_quadrants_angle.append(angle_dist)
            else:

                pass

        if len(right_quadrants_idx) > 0:
            if len(right_quadrants_angle) == 1:
                angle_dist = right_quadrants_angle[0]
                if angle_dist > np.deg2rad(single_angle_max):
                    # print(f"only oneright angle dist: {angle_dist}")
                    right_idx = -1
                else:
                    right_idx = right_quadrants_idx[0]
            elif len(right_quadrants_angle) > 1:
                right_idx = right_quadrants_idx[np.argmin(right_quadrants_dist)]
            else:
                right_idx = -1

            # right_idx = right_quadrants_idx[np.argmin(right_quadrants_dist)]
        else:
            # print(f"no right neighbours for point {i}")
            right_idx = -1

        if len(upper_quadrants_idx) > 0:
            if len(upper_quadrants_angle) == 1:
                angle_dist = upper_quadrants_angle[0]
                if angle_dist > np.deg2rad(single_angle_max):
                    # print(f"only one upper angle dist: {angle_dist}")
                    upper_idx = -1
                else:
                    upper_idx = upper_quadrants_idx[0]
            elif len(upper_quadrants_angle) > 1:
                upper_idx = upper_quadrants_idx[np.argmin(upper_quadrants_dist)]
            else:
                upper_idx = -1
        else:
            # print(f"no upper neighbours for point {i}")
            upper_idx = -1

        if len(left_quadrants_idx) > 0:
            if len(left_quadrants_angle) == 1:
                angle_dist = left_quadrants_angle[0]
                if angle_dist > np.deg2rad(single_angle_max):
                    print(f"only one left angle dist: {angle_dist}")
                    left_idx = -1
                else:
                    left_idx = left_quadrants_idx[0]
            elif len(left_quadrants_angle) > 1:
                left_idx = left_quadrants_idx[np.argmin(left_quadrants_dist)]
            else:
                left_idx = -1
        else:
            # print(f"no left neighbours for point {i}")
            left_idx = -1

        if len(lower_quadrants_idx) > 0:
            if len(lower_quadrants_angle) == 1:
                angle_dist = lower_quadrants_angle[0]
                if angle_dist > np.deg2rad(single_angle_max):
                    print(f"only one lower angle dist: {angle_dist}")
                    lower_idx = -1
                else:
                    lower_idx = lower_quadrants_idx[0]
            elif len(lower_quadrants_angle) > 1:
                lower_idx = lower_quadrants_idx[np.argmin(lower_quadrants_dist)]
            else:
                lower_idx = -1
        else:
            # print(f"no lower neighbours for point {i}")
            lower_idx = -1
        neighor_idx = np.array([right_idx, upper_idx, left_idx, lower_idx])
        neighbor_idxs.append(neighor_idx)

    neighbor_idxs = np.array(neighbor_idxs)
    return neighbor_idxs

from scipy.spatial import cKDTree

# ----------------------------------------------------------------------
def connected_components_from_neighbours(
        neighbor_idxs: np.ndarray,
        points:        np.ndarray,
        close_tol: float = 0.0      # ≤ 0  ⇒ no extra proximity edges
):
    """
    neighbour_idxs : (N,k)      ; neighbour_idxs[i,0] is 'horizontal' pointer,
                                  neighbour_idxs[i,1] optional vertical pointer.
    points         : (N,d)      ; coordinates
    close_tol      : float      ; add an edge when two points are within
                                 this Euclidean distance.

    Returns exactly the same tuple structure as the original version,
    but with extra proximity‑based connections included.
    """
    if len(points) == 0:
        return (np.array([]), [], np.array([]), [], [], [])

    N      = len(points)
    dmaskH = np.zeros((N, N), bool)       # horizontal adjacency
    dmaskV = np.zeros((N, N), bool)       # vertical   adjacency

    # -- explicit graph edges from neighbour_idxs ------------------------
    for i in range(N):
        if neighbor_idxs[i, 0] != -1:
            j = neighbor_idxs[i, 0]
            dmaskH[i, j] = dmaskH[j, i] = True
        if neighbor_idxs.shape[1] > 1 and neighbor_idxs[i, 1] != -1:
            j = neighbor_idxs[i, 1]
            dmaskV[i, j] = dmaskV[j, i] = True

    # -- proximity‑based extra edges ------------------------------------
    if close_tol > 0:
        tree = cKDTree(points)
        pairs = tree.query_pairs(r=close_tol)      # unordered set of (u,v)
        for u, v in pairs:
            dmaskH[u, v] = dmaskH[v, u] = True     # count for both graphs
            dmaskV[u, v] = dmaskV[v, u] = True

    # -- generic DFS connected‑component helper -------------------------
    def _components(adj):
        labels = np.full(N, -1, int)
        comps, lbl = [], 0
        for i in range(N):
            if labels[i] != -1:
                continue
            stack, comp = [i], []
            labels[i] = lbl
            while stack:
                n = stack.pop()
                comp.append(n)
                neigh = np.where(adj[n])[0]
                for j in neigh:
                    if labels[j] == -1:
                        labels[j] = lbl
                        stack.append(j)
            comps.append(comp)
            lbl += 1
        return labels, comps

    labels_H, comps_H = _components(dmaskH)
    labels_V, comps_V = _components(dmaskV)

    # only keep non‑trivial point arrays
    horiz_pts = [points[c] for c in comps_H if len(c) > 1]
    vert_pts  = [points[c] for c in comps_V if len(c) > 1]

    return labels_H, comps_H, labels_V, comps_V, horiz_pts, vert_pts

def save_crops(slx, sly, slab_size, file_name, slab, crop, output_path='.'):
    """
    Save image crop data to a numpy file.
    
    Parameters
    ----------
    slx, sly : int
        Crop location indices.
    slab_size : int
        Size of the slab.
    file_name : str
        Base name for output file.
    slab : ndarray
        Slab data (currently unused).
    crop : ndarray
        The cropped image data to save.
    output_path : str, optional
        Directory path for output file (default: '.').
        
    Returns
    -------
    dict
        Dictionary containing all saved crops including the new one.
    """
    import os
    crop_file_name = f'{output_path}/{file_name}_crops_new.npy'
    if os.path.exists(crop_file_name):
        crops = np.load(crop_file_name, allow_pickle=True).item()
    else:
        crops = {}
    len_crops = len(crops)
    crops[len_crops] = {}
    crops[len_crops]['crop'] = crop
    crops[len_crops]['slx'] = slx
    crops[len_crops]['sly'] = sly
    crops[len_crops]['slab_size'] = slab_size
    crops[len_crops]['file_name'] = file_name
    np.save(crop_file_name, crops)
    return crops




import numpy as np
from collections import defaultdict

# ------------------------------------------------------------------
# neighbour order in neighbor_idxs = [right, up, left, down]
RIGHT, UP, LEFT, DOWN = range(4)
DIR_PAIRS = [(RIGHT, LEFT), (UP, DOWN)]          # edges to make graph undirected
# ------------------------------------------------------------------

def connected_components_from_neighbours(neighbor_idxs: np.ndarray, points: np.ndarray, prox_tol = 0):
    """
    Parameters
    ----------
    neighbor_idxs : (N, 4) int array
        Output of your `horizontal_link_clustering`:
        -1 where no neighbour, otherwise index of neighbour point.
        neighbour order must be  [right, up, left, down].

    Returns
    -------
    comp_labels : (N,) int array
        Component ID for every point (–1 if isolated).
    components : list[list[int]]
        List of components; each component is a list of point indices.
    """

    N = len(neighbor_idxs)
    parent_horizontal = np.arange(N)        # union–find parent pointers
    parent_vertical = np.arange(N)        # union–find parent pointers

    # --- union–find helpers ------------------------------------------------
    def find_horizontal(a):
        while parent_horizontal[a] != a:
            parent_horizontal[a] = parent_horizontal[parent_horizontal[a]]
            a = parent_horizontal[a]
        return a

    def union_horizontal(a, b):
        ra, rb = find_horizontal(a), find_horizontal(b)
        if ra != rb:
            parent_horizontal[rb] = ra
    def find_vertical(a):
        while parent_vertical[a] != a:
            parent_vertical[a] = parent_vertical[parent_vertical[a]]
            a = parent_vertical[a]
        return a

    def union_vertical(a, b):
        ra, rb = find_vertical(a), find_vertical(b)
        if ra != rb:
            parent_vertical[rb] = ra

    # ----------------------------------------------------------------------
    # build an undirected graph: for each directed edge i→j
    # also union the opposite direction j→i, when present
    for i in range(N):
        for d_idx, d_opp in DIR_PAIRS[:1]:          # right‑left, up‑down
            j = neighbor_idxs[i, d_idx]
            if j == -1:
                pass
            else:
                union_horizontal(i, j)
            j = neighbor_idxs[i, d_opp]
            if j == -1:
               pass 
            else:
                union_horizontal(j, i)
                
        for d_idx, d_opp in DIR_PAIRS[1:]:          # right‑left, up‑down
            j = neighbor_idxs[i, d_idx]
            if j == -1:
                pass
            else:
                union_vertical(i, j)
            j = neighbor_idxs[i, d_opp]
            if j == -1:
                pass
            else:
                union_vertical(j, i)

    if prox_tol > 0:
        xy = points[:, :]                # use (x,y) only
        tree = cKDTree(xy)
        for i in range(N):
            for j in tree.query_ball_point(xy[i], r=prox_tol):
                if j == i:
                    continue
                union_horizontal(i, j)     # make sure same row component
                union_vertical(i, j)       # and same column component

    comp_map_horizontal = defaultdict(list)
    comp_map_vertical = defaultdict(list)
    for i in range(N):
        root = find_horizontal(i)
        comp_map_horizontal[root].append(i)
        root = find_vertical(i)
        comp_map_vertical[root].append(i)

    # produce outputs
    comp_labels_horizontal = np.full(N, -1, int)
    for new_id, (root, idxs) in enumerate(comp_map_horizontal.items()):
        for idx in idxs:
            comp_labels_horizontal[idx] = new_id
    components_horizontal = list(comp_map_horizontal.values())

    comp_labels_vertical = np.full(N, -1, int)
    for new_id, (root, idxs) in enumerate(comp_map_vertical.items()):
        for idx in idxs:
            comp_labels_vertical[idx] = new_id
    components_vertical = list(comp_map_vertical.values())
    horizontal_points = [points[comp] for comp in components_horizontal]
    vertical_points = [points[comp] for comp in components_vertical]

    return comp_labels_horizontal, components_horizontal, comp_labels_vertical, components_vertical, horizontal_points, vertical_points


import numpy as np
from collections import defaultdict



from scipy.spatial import cKDTree          # fast KD‑tree
from typing import List

def _collapse_duplicates(lines: List[np.ndarray],
                         axis: int,
                         min_sep: float) -> List[np.ndarray]:
    """
    axis = 1 ➜ use median y (horizontal filtering)
    axis = 0 ➜ use median x (vertical   filtering)
    """
    if min_sep <= 0 or len(lines) < 2:
        return lines                      # nothing to filter

    keyvals = [(i, float(np.median(line[:, axis])))
               for i, line in enumerate(lines)]
    keyvals.sort(key=lambda t: t[1])      # sort by coordinate

    kept, coords_kept = [], []
    for idx, coord in keyvals:
        # if all(abs(coord - c) >= min_sep for c in coords_kept):
        kept.append(lines[idx])
            # coords_kept.append(coord)
    return kept


# --------------------------------------------------------------------
# 1)  build the grid after both filters
# --------------------------------------------------------------------
def build_grid(
        vertical: List[np.ndarray],
        horizontal:   List[np.ndarray],
        tol_upper_bound: float           = 1.0,
        tol_lower_bound: float           = 1.0,
        *,
        min_intra_h: float = 0.0,
        min_intra_v: float  = 0.0,
        pick: str = "median"
) -> np.ndarray:
    """
    Construct a regular grid from detected horizontal and vertical line segments.
    
    This function matches intersection points between horizontal and vertical lines
    to create a structured grid representation. It handles missing intersections
    and selects optimal representatives when multiple candidates exist.
    
    Parameters
    ----------
    vertical : list of ndarray
        List of vertical line segments, each shape (N_i, D) where D is 2 or 3.
    horizontal : list of ndarray
        List of horizontal line segments, each shape (N_j, D).
    tol_upper_bound : float, optional
        Maximum distance for matching intersection points (default: 1.0).
    tol_lower_bound : float, optional
        Minimum distance threshold (currently unused, default: 1.0).
    min_intra_h : float, optional
        Minimum separation between points within horizontal lines (default: 0.0).
    min_intra_v : float, optional
        Minimum separation between points within vertical lines (default: 0.0).
    pick : str, optional
        Method for selecting representative from clustered points:
        'median', 'mean', or 'first' (default: 'median').
    
    Returns
    -------
    grid : ndarray, shape (n_rows, n_cols, D)
        Regular grid where grid[i, j] contains intersection coordinates.
        NaN values indicate missing intersections.
    vertical : list of ndarray
        Processed vertical lines after thinning.
    horizontal : list of ndarray
        Processed horizontal lines after thinning.
    """


    vertical = [_thin_vertices(h, axis=2, min_sep=min_intra_v, pick=pick)
         for h in vertical]
    horizontal = [_thin_vertices(v, axis=1, min_sep=min_intra_h, pick=pick)
         for v in horizontal]
    vertical = _collapse_duplicates(vertical, axis=1, min_sep=min_intra_v/2)
    horizontal = _collapse_duplicates(horizontal, axis=2, min_sep=min_intra_h/2)

    n_rows, n_cols = len(vertical), len(horizontal)
    D      = vertical[0].shape[1]
    grid   = np.full((n_rows, n_cols, D), np.nan)

    # -- 2. stack all vertices from vertical lines -----------------------
    V_xy, V_owner = [], []
    for j, vline in enumerate(vertical):
        # vline = vline[np.argsort(vline[:,1])]
        V_xy.append(vline[:, :])         # use x,y for KD‑tree
        V_owner.extend([j] * len(vline))
    V_xy    = np.vstack(V_xy)
    V_owner = np.asarray(V_owner)
    tree_V    = cKDTree(V_xy)

    H_xy, H_owner = [], []
    for i, hline in enumerate(horizontal):
        H_xy.append(hline[:, :])
        H_owner.extend([i] * len(hline))
    H_xy = np.vstack(H_xy)
    H_owner = np.asarray(H_owner)
    tree_H = cKDTree(H_xy)

    # -- 3. query KD‑tree for each horizontal vertex ---------------------
    for i, vline in enumerate(vertical):
        # make sure hline is sorted
        # vline = vline[np.argsort(vline[:,2])]
        success = False
        for p_ind, p in enumerate(vline):
            dist, idx = tree_H.query(p[:], distance_upper_bound=tol_upper_bound)
            if not np.isfinite(dist):     # no match within tol
                continue
            j = H_owner[idx]
            # check if grid[i, j] is nan
            if np.isnan(grid[i, j, 0]):
                grid[i, j] = p                # store entire point (2D/3D)
                success = True
            else:
                # print(f'conflict at {i}, {j}')
                old_dist, old_idx = tree_V.query(grid[i, j, :], distance_upper_bound=tol_upper_bound)
                if dist < old_dist:
                    grid[i, j] = p
                success = True
            # check if any nans left in grid[i,:,0]
            # if success:
            #     if np.isnan(grid[i,:,0]).any():
            #         # print(f"nans left in grid[i,:,0]")
            #         if j>0:
            #             if np.isnan(grid[i,j-1,0]):
            #                 if p_ind-1>-1:
            #                     if np.abs(vline[p_ind-1,0] - vline[p_ind,0]) > tol_lower_bound:
            #                         grid[i,j-1,:] = vline[p_ind-1]
            #                         print(f"fillin grid[i,j-1,:] = vline[p_ind-1]")
            #         if j<n_cols-1:
            #             if np.isnan(grid[i,j+1,0]):
            #                 if p_ind+1<len(vline):
            #                     if np.abs(vline[p_ind+1,0] - vline[p_ind,0]) > tol_lower_bound:
            #                         grid[i,j+1,:] = vline[p_ind+1]
            #                         print(f"fillin grid[i,j+1,:] = vline[p_ind+1]")
              

            

    for j, hline in enumerate(horizontal):
        # hline = hline[np.argsort(hline[:, 1])]
        success = False
        for p in hline:
            dist, idx = tree_V.query(p[:], distance_upper_bound=tol_upper_bound)
            if not np.isfinite(dist):
                continue
            i = V_owner[idx]
            if np.isnan(grid[i, j, 0]):          # fill only if still empty
                grid[i, j] = p
            else:
                old_dist, old_idx = tree_H.query(grid[i, j, :], distance_upper_bound=tol_upper_bound)
                if dist < old_dist:
                    grid[i, j] = p

            # if success:
            #     if np.isnan(grid[:,j,0]).any():
            #         if i>0:
            #             if np.isnan(grid[i-1,j,0]):
            #                 if p_ind-1>-1:
            #                     if np.abs(hline[p_ind-1,0] - hline[p_ind,0]) > tol_lower_bound:
            #                         grid[i-1,j,:] = hline[p_ind-1]
            #                         print(f"fillin grid[i-1,j,:] = hline[p_ind-1]")
            #         if i<n_rows-1:
            #             if np.isnan(grid[i+1,j,0]):
            #                 if p_ind+1<len(hline):
            #                     if np.abs(hline[p_ind+1,0] - hline[p_ind,0]) > tol_lower_bound:
            #                         grid[i+1,j,:] = hline[p_ind+1]
            #                         print(f"fillin grid[i+1,j,:] = hline[p_ind+1]")





               
        # find if any point have been inserted into the grid
        # if there are any nans, then check if there are point in vertical[j] that are above or below the nan point

    return grid, vertical, horizontal


def _thin_vertices(line: np.ndarray,
                   axis: int,
                   min_sep: float,
                   pick: str = "median") -> np.ndarray:
    """
    Remove in‑line duplicates that sit closer than `min_sep` along `axis`.
      axis = 0 ➜ x  (for vertical line)
      axis = 1 ➜ y  (for horizontal line)

    Returns the same dimensionality as input (2‑D or 3‑D points).
    """
    if min_sep <= 0 or len(line) < 2:
        return line

    # sort by the collapse axis
    sort_idx = np.argsort(line[:, axis])
    pts = line[sort_idx]

    clusters, current = [], [pts[0]]
    for p in pts[1:]:
        if abs(p[axis] - current[-1][axis]) < min_sep:
            # print(f"p: {p}, current: {current}")
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)

    # pick a single representative per cluster
    if pick == "median":
        reps = [np.median(c, axis=0) for c in clusters] 
    elif pick == "first":
        reps = [c[0] for c in clusters]
    elif pick == "mean":
        reps = [np.mean(c, axis=0) for c in clusters]
    else:
        raise ValueError("pick must be 'median', 'first', or 'mean'")
    
    return np.vstack(reps)

def rung_spacings(grid: np.ndarray):
    """
    Parameters
    ----------
    grid : (rows, cols, D)  array
        D = 2 or 3 coordinates.  NaNs mark missing intersections.

    Returns
    -------
    d_vert : (rows‑1, cols)        spacing between successive rows
    d_horz : (rows,   cols‑1)      spacing between successive columns
    """
    # validity mask: True where all coordinates are finite
    valid = np.isfinite(grid).all(axis=2)

    # -------- vertical (row‑to‑row) distances ---------------------
    diff_v = grid[1:] - grid[:-1]                    # shape (r‑1, c, D)
    d_vert = np.linalg.norm(diff_v, axis=2)          # Euclidean norm
    bad_v  = ~(valid[1:] & valid[:-1])               # any NaN in pair
    d_vert[bad_v] = np.nan

    # -------- horizontal (col‑to‑col) distances -------------------
    diff_h = grid[:, 1:] - grid[:, :-1]              # shape (r, c‑1, D)
    d_horz = np.linalg.norm(diff_h, axis=2)
    bad_h  = ~(valid[:, 1:] & valid[:, :-1])
    d_horz[bad_h] = np.nan

    return d_vert, d_horz



def merge_lines(horizontal_lines, tol = 100, max_horizontal_lines = 5):
    """
    Merge horizontal lines that are spatially close to each other.
    
    This function combines line segments that are nearby in the coordinate space,
    reducing redundant detections and improving grid structure.
    
    Parameters
    ----------
    horizontal_lines : list of ndarray
        List of horizontal line segments to potentially merge.
    tol : float, optional
        Distance tolerance for merging lines (default: 100).
    max_horizontal_lines : int, optional
        Only attempt merging if more than this many lines exist (default: 5).
        
    Returns
    -------
    list of ndarray
        Merged list of horizontal lines with redundant segments combined.
    """
    # if the start and end of the horizontal line are close to the start and end of another horizontal line (in terms of y axis), then merge the two lines
    if len(horizontal_lines) > max_horizontal_lines:
        for i in range(len(horizontal_lines)):
            for j in range(len(horizontal_lines)):
                if i != j:
                    if len(horizontal_lines[i]) == 0 or len(horizontal_lines[j]) == 0:
                        continue
                    else:
                        first_line = horizontal_lines[i]
                        second_line = horizontal_lines[j]
                        # check if first and last point are closer or last and first point are closer
                        mean_first_line = np.mean(first_line[:,1])
                        mean_second_line = np.mean(second_line[:,1])
                        if mean_first_line > mean_second_line:
                            right_point = first_line[-1,:] 
                            left_point = second_line[0,:]
                        if mean_first_line < mean_second_line:
                            left_point = first_line[-1,:] 
                            right_point = second_line[0,:]
                        if np.abs(left_point[2] - right_point[2]) < tol:
                            print(f"merging lines {i} and {j}")
                            horizontal_lines[i] = np.vstack((first_line, second_line))
                            horizontal_lines[j] = np.array([])
                            # break
    horizontal_lines = [line for line in horizontal_lines if line.size > 0]
    return horizontal_lines




# ============================================================================
# MASK PROCESSING UTILITIES
# ============================================================================

from scipy.spatial import cKDTree
from skimage.draw import line_nd         # tiny helper in scikit‑image
from scipy import ndimage as ndi

def fill_mask(mask, extensionz = 1, extensionxy = 50):
    """
    Fill and extend a 3D binary mask around detected points.
    
    This function expands regions around points in a 3D mask and fills internal
    cavities to create a continuous volume.
    
    Parameters
    ----------
    mask : ndarray, shape (Z, Y, X)
        3D binary mask to process.
    extensionz : int, optional
        Extension distance in Z direction (default: 1).
    extensionxy : int, optional
        Extension distance in X and Y directions (default: 50).
        
    Returns
    -------
    ndarray
        Filled and extended binary mask with same shape as input.
    """

    pts = np.vstack(np.nonzero(mask)).T       # N×3 [z,y,x] coords
    dz, dy, dx = mask.shape
    for ind, pt in enumerate(pts):
        zz, yy, xx = pt
        zlims = [max(0, zz-extensionz), min(dz, zz+extensionz)]
        ylims = [max(0, yy-extensionxy), min(dy, yy+extensionxy)]
        xlims = [max(0, xx-extensionxy), min(dx, xx+extensionxy)]
        mask[zlims[0]:zlims[1], ylims[0]:ylims[1], xlims[0]:xlims[1]] = 1

    # Finally flood‑fill internal cavities
    filled = ndi.binary_fill_holes(mask)
    return filled.astype(mask.dtype)        
