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
import sys
import tifffile
import os
# print("path", path)
path = "/groups/ahrens/home/ruttenv/python_packages/mesoscope/"
sys.path.append(path)
from mesoscope.utils import volume_converter
from mesoscope.utils import photomap_utils 
#from scipy import ndimage as ndi
from importlib import reload
from scipy.ndimage import maximum_filter
from scipy.ndimage import binary_dilation, binary_erosion
import warnings
from pathlib import Path
from importlib import reload

# === CONFIGURATION ===
image_path =  '/nrs/ahrens/Virginia_nrs/wbi_paper_data/photomap/250730_exm_photomap/post/sagittal/composite_image_250730_sagittal_post_db_00001_TZCXY_ds3.tif'     # 3D image, shape (Z, Y, X)

results_path   = '/nrs/ahrens/Virginia_nrs/wbi_paper_data/photomap/250730_exm_photomap/post/sagittal/points/'
# === LOAD DATA ===
#%%
data = tf.imread(image_path)

templates = np.load(results_path + 'templates.npy', allow_pickle=True).item()
#%%


#%% EXTRACT GRID POINTS
from mesoscope.utils import photomap_utils 
from time import time
reload(photomap_utils)

results = {}
for plane, subdata in enumerate(data):
    start_time = time()
    results[plane] = {}
    for ind, key in enumerate(templates):
        results[plane][ind] = {}
        if ind <1:
            print(f"Processing {plane}{key}")
            current_crop = templates[key]
            dx = np.min(current_crop.shape)
            current_crop = current_crop[:dx, :dx]
            C = photomap_utils.zncc(subdata, current_crop)
            C_mask = photomap_utils.local_top_percentile(C, patch=80, percentile=90)
            centers, LoG = photomap_utils.find_blob_centers(C,
                                        radius_px=8,   
                                        min_distance=50,
                                        threshold_rel=0.6)
            results[plane][ind]['centers'] = centers
            results[plane][ind]['LoG'] = LoG
            results[plane][ind]['C_mask'] = C_mask
            results[plane][ind]['C'] = C
    end_time = time()
    print(f"Time taken for plane {plane}: {(end_time - start_time):.2f} seconds")


#%% LOAD NAPARI LAYERS WITH INITIAL LABELS
from mesoscope.napari import napari_utils
reload(napari_utils)
folders = sorted(glob(results_path + '/napari_layers_with_labels_sam*/'))
latest_folder = folders[-1]
files = sorted(glob(latest_folder + '/*.pkl'))
file = files[0]

layers = napari_utils.load_layers_by_type(file, show_viewer=False, allowed_types=['Points'])
columns_ind = 0
rows_ind = 1
plane_ind = 2
onsample_ind = 3
points = layers[list(layers.keys())[columns_ind]]['data']
unique_colors_rows, reverse_indices_rows, counts_rows = np.unique(layers[list(layers.keys())[rows_ind]]['face_color'], axis=0, return_inverse=True, return_counts=True)
unique_colors_columns, reverse_indices_columns, counts_columns = np.unique(layers[list(layers.keys())[columns_ind]]['face_color'], axis=0, return_inverse=True, return_counts=True)
unique_colors_planes, reverse_indices_planes, counts_planes = np.unique(layers[list(layers.keys())[plane_ind]]['face_color'], axis=0, return_inverse=True, return_counts=True)
unique_colors_onsample, reverse_indices_onsample, counts_onsample = np.unique(layers[list(layers.keys())[onsample_ind]]['face_color'], axis=0, return_inverse=True, return_counts=True)
num_rows = len(unique_colors_rows)
num_columns = len(unique_colors_columns)
num_planes = len(unique_colors_planes)
num_onsamples = len(unique_colors_onsample)
print(f"number of rows: {len(unique_colors_rows)}, number of columns: {len(unique_colors_columns)}, number of planes: {len(unique_colors_planes)}, number of onsamples: {len(unique_colors_onsample)}")
# print(f"counts rows: {counts_rows}, \ncounts columns: {counts_columns}, \ncounts planes: {counts_planes}")

#%% FIND SMALL CLUSTERS to fix them manually
small_cluster_indices = np.where(counts_onsample < 100)[0]

if len(small_cluster_indices) > 0:
    # Find all points belonging to ANY small cluster
    small_points_mask = np.isin(reverse_indices_planes, small_cluster_indices)
    small_points = points[small_points_mask]
    print(f"Found {len(small_points)} points in {len(small_cluster_indices)} small clusters")
else:
    print("No clusters with <10 points found")

#%% SAVE finetuned LABELS
results = {}
results['points'] = points
results['row_labels'] = reverse_indices_rows
results['column_labels'] = reverse_indices_columns
results['plane_labels'] = reverse_indices_planes
results['onsample_labels'] = reverse_indices_onsample
np.save(results_path + 'results_points_labels_final.npy', results)
#%%
results = np.load(results_path + 'results_points_labels_final.npy', allow_pickle=True).item()
points = results['points']
row_labels = results['row_labels']
column_labels = results['column_labels']
plane_labels = results['plane_labels']
onsample_labels = results['onsample_labels']
unique_colors_rows = np.unique(row_labels)
unique_colors_columns = np.unique(column_labels)
unique_colors_planes = np.unique(plane_labels)

#%%

def remap_labels(labels, points, axis = 1):
    unique_labels = np.unique(labels)
    # sort the groups by their median z value and label them in that order
    all_zs = []
    for label in unique_labels:
        indices_ = np.argwhere(labels==label)[:,0]
        subpoints = points[indices_]
        z_values = subpoints[:,axis]
        median_z = np.median(z_values)
        all_zs.append(median_z)
    all_zs = np.array(all_zs)
    order = np.argsort(all_zs)
    new_labels = np.zeros_like(labels)
    for ind, order_ind in enumerate(order):
        inds_ = np.argwhere(labels==unique_labels[order_ind])[:,0]
        new_labels[inds_] = unique_labels[ind]
    return new_labels

labels  = row_labels
new_row_labels = remap_labels(labels, points, axis = 1)

labels = column_labels
new_column_labels = remap_labels(labels, points, axis = 2)

labels = plane_labels
new_plane_labels = remap_labels(labels, points, axis = 0)

#%% REORGANIZE POINTS INTO GRID
num_rows = len(unique_colors_rows)
num_columns = len(unique_colors_columns)
num_planes = len(unique_colors_planes)  
grid_points = np.nan*np.ones((num_planes, num_rows, num_columns, 3))
grid_labels = np.zeros((num_planes, num_rows, num_columns))
grid_row_labels = np.zeros((num_planes, num_rows, num_columns))
grid_column_labels = np.zeros((num_planes, num_rows, num_columns))
grid_points[new_plane_labels, new_row_labels, new_column_labels] = points
grid_labels[new_plane_labels, new_row_labels, new_column_labels] = onsample_labels
grid_row_labels[new_plane_labels, new_row_labels, new_column_labels] = new_row_labels
grid_column_labels[new_plane_labels, new_row_labels, new_column_labels] = new_column_labels
#%%
grid_results = {}
grid_results['points'] = grid_points
grid_results['labels'] = grid_labels
grid_results['row_labels'] = grid_row_labels
grid_results['column_labels'] = grid_column_labels
grid_results['row_labels'] = new_row_labels
grid_results['napari_points'] = new_column_labels
#%%
napari_points = grid_points.copy()
napari_points[:,:,:,0] = np.arange(grid_points.shape[0])[:,None,None]
napari_points = napari_points.reshape(-1,3)
grid_results['napari_points'] = napari_points
napari_row_labels = grid_row_labels.reshape(-1)
napari_column_labels = grid_column_labels.reshape(-1)
napari_fish_labels = grid_labels.reshape(-1)
grid_results['napari_row_labels'] = napari_row_labels
grid_results['napari_column_labels'] = napari_column_labels
grid_results['napari_fish_labels'] = napari_fish_labels

np.save(results_path + 'grid_results_napari.npy', grid_results)



#%%
subpoints = grid_points[8][:,:,:]
row = 10
column = 10
pl.figure(figsize=(5,3))
for ind,row in enumerate(range(num_rows)):
    color = pl.get_cmap('magma')(ind/num_rows)
    pl.plot(subpoints[row,:,1],subpoints[row,:,2],'o-',alpha=1, color = color)
pl.figure(figsize=(5,3))
for ind,column in enumerate(range(num_columns)):
    color = pl.get_cmap('magma')(ind/num_columns)
    pl.plot(subpoints[:,column,1],subpoints[:,column,2],'o-',alpha=1, color = color)


#%% ############ FLATTEN IMAGE ############
planes = []
for plane_ind in range(num_planes):
# for plane_ind in [13]:
    dummy_vol = np.zeros_like(data)
    delta_xy = 250
    delta_z = 0

    pointsz_ = grid_points[plane_ind, :, :, 0]
    pointsy_ = grid_points[plane_ind, :, :, 1]
    pointsx_ = grid_points[plane_ind, :, :, 2]

    # Flatten and filter out nan points
    valid_mask = ~np.isnan(pointsz_) & ~np.isnan(pointsy_) & ~np.isnan(pointsx_)
    zs = pointsz_[valid_mask].astype(int)
    ys = pointsy_[valid_mask].astype(int)
    xs = pointsx_[valid_mask].astype(int)

    for z, y, x in zip(zs, ys, xs):
        zmin, zmax = max(z - delta_z, 0), min(z + delta_z+1 , dummy_vol.shape[0])
        ymin, ymax = max(y - delta_xy, 0), min(y + delta_xy + 1, dummy_vol.shape[1])
        xmin, xmax = max(x - delta_xy, 0), min(x + delta_xy + 1, dummy_vol.shape[2])
        dummy_vol[zmin:zmax, ymin:ymax, xmin:xmax] = 1

    # max filter in xy
    # from scipy.ndimage import maximum_filter
    # dummy_vol_max = np.array([maximum_filter(dummy_vol_, size=50) for dummy_vol_ in dummy_vol])

    plane_with_most_points = np.argmax(dummy_vol.sum(axis = (1,2)))
    print(plane_with_most_points)

    dummy_vol[plane_with_most_points,400:1600,1200:1800] = 1
    coverage_mask = dummy_vol.sum(axis=0)  # shape: (Y, X)

    if np.any(coverage_mask == 0):
        print('filling in missing data')
        # Find the plane with the most data (already computed)
        dummy_vol[plane_with_most_points, coverage_mask == 0] = 1
    vol = data.copy()
    vol[dummy_vol ==0] = np.nan
    vol = np.nanmin(vol, axis = 0)
    planes.append(vol)
planes = np.stack(planes, axis = 0)


planes_tzcxy = planes[None,:,None].astype(np.float32)

with tifffile.TiffWriter(results_path + 'planes3.tif', imagej=True) as tif:
    tif.write(planes_tzcxy,resolution=(1.0, 1.0))

#%%
pl.figure(figsize=(5,5))
pl.imshow(planes[-1][::4,::4], cmap='gray')
pl.colorbar()
#%%
# planes = np.stack(planes, axis = 0)
#%%
for plane_ind in range(num_planes):
    pl.figure(figsize=(5,5))
    pl.title(f"plane {plane_ind}")
    pl.imshow(planes[plane_ind][::4,::4], cmap='gray')
    # pl.colorbar()
    
    pl.axis('off')
    pl.tight_layout()

#%%


#%%
column = 30
diff_axis2 = np.diff(subpoints[:,:,2], axis = 1).flatten()
diff_axis1 = np.diff(subpoints[:,:,1], axis = 0).flatten()

diff_axis1 = diff_axis1[~np.isnan(diff_axis1)]
diff_axis2 = diff_axis2[~np.isnan(diff_axis2)]
pl.figure(figsize=(5,2))
pl.hist(diff_axis2, bins = 30)
# pl.figure(figsize=(5,2))
pl.hist(diff_axis1, bins = 30)
pl.ylim(0, 100)

#%%

#%%
pl.figure(figsize=(5,3))
for plane_ind in range(num_planes):
    subpoints = grid_points[plane_ind][:,:,:]
    for ind, column in enumerate(range(num_columns)):
        color = gridiness[plane_ind, :,column]
        min_ = 0.3
        color = (color - min_)/(1 - min_)
        color = pl.get_cmap('copper')(color)
        pl.scatter(subpoints[:,column,1],subpoints[:,column,2], color = color, s = 3)
        # pl.plot(subpoints[:,column,1],subpoints[:,column,2],'-', color = 'black', zorder = -1, lw = 1)
    pl.axis('off')
#%%
plane_ind = 10
subpoints = grid_points[plane_ind][:,:,:]
pl.figure(figsize=(5,3))
for ind, column in enumerate(range(num_columns)):
    color = grid_labels[plane_ind, :,column]
    color = pl.get_cmap('copper')(color)
    pl.scatter(subpoints[:,column,1],subpoints[:,column,2], color = color, s = 3)
    # pl.plot(subpoints[:,column,1],subpoints[:,column,2],'-', color = 'black', zorder = -1, lw = 1)
pl.axis('off')
#%%




#%%
pl.figure(figsize=(5,10))
pl.imshow(upsampled_mask.T[::4,::4]>0.47, cmap='gray')
pl.colorbar(shrink=0.5)
pl.clim(0, 1)



#%%

pl.figure(figsize=(5,10))
pl.imshow(upsampled_mask[ind].T, cmap='gray')
pl.colorbar(shrink=0.5)
pl.clim(0, 1)
pl.figure(figsize=(5,10))
pl.imshow(mask_[ind].T, cmap='gray')
pl.colorbar(shrink=0.5)
pl.clim(0, 1)
#%%





#%%
mask = subdata_max < 130
subdata_max = maximum_filter(subdata, size=17)
mask = subdata_max < 130
from scipy.ndimage import binary_dilation, binary_erosion
# mask = binary_dilation(mask, iterations=30)
# mask = binary_erosion(mask, iterations=10)

pl.figure(figsize=(8,5))
pl.imshow(subdata_max[::4,::4].T, cmap='gray')
pl.colorbar(shrink=0.5)
pl.figure(figsize=(5,10))
pl.imshow(mask[::4,::4].T, cmap='gray')

#%%
pl.figure(figsize=(5,10))
pl.imshow(data_min[::4,::4].T, cmap='gray')
#%%
cutoff = 40

mask = data[plane_ind] < cutoff
pl.figure(figsize=(5,10))
pl.imshow(mask[::4,::4], cmap='gray')
# pl.hist(data[::10,::4,::4].flatten())


#%%


row_ind = 20
column_ind = 19
subpoints_um = points[(row_labels==row_ind)]
subpoints_um = points[(column_labels==column_ind)]


pl.figure(figsize=(5,10))
pl.plot(subpoints_um[:,0],subpoints_um[:,1],'o',alpha=0.5)
#%%
from mesoscope.utils import point_deduplication
reload(point_deduplication)
points_um = np.copy(points)
points_um[:,0] = points_um[:,0]

all_unique_points = []
all_selected_indices = []
for row_ind in range(num_rows):
    locs = np.argwhere(reverse_indices_rows==row_ind)[:,0]
    subpoints_um = points_um[locs]

    x_y_tolerance = 1.9
    z_tolerance = 1.0
    unique_points, selected_indices, mapping = point_deduplication.merge_duplicate_points(
        subpoints_um, 
        xy_tolerance=x_y_tolerance,  # Max 10 pixel difference in x-y
        z_tolerance=z_tolerance,     # Max 1 unit difference in z,
        return_mapping=True, 
        merge_method='min_z',
        return_indices=True
    )
    all_unique_points.append(unique_points)
    all_selected_indices.append(locs[selected_indices])
all_unique_points = np.concatenate(all_unique_points)
all_selected_indices = np.concatenate(all_selected_indices)
#%%
row_labels_deduplicated = row_labels[all_selected_indices]
column_labels_deduplicated = column_labels[all_selected_indices]
results['unique_points'] = all_unique_points
results['selected_indices'] = all_selected_indices
results['row_labels_deduplicated'] = row_labels_deduplicated
results['column_labels_deduplicated'] = column_labels_deduplicated
np.save(results_path + 'results_points_labels_deduplicated.npy', results)
#%%
plane_labels_deduplicated = np.zeros_like(row_labels_deduplicated)

for row_ind in range(num_rows):
    indices = np.argwhere(row_labels_deduplicated==row_ind)[:,0]
    subpoints_um = all_unique_points[indices]
    tmp = subpoints_um.copy()
    tmp[:,0] = tmp[:,0]*350
    from sklearn.cluster import DBSCAN
    np.random.seed(42)
    db = DBSCAN(eps=400, min_samples=2, metric = 'euclidean').fit(tmp[:,:])
    labels = db.labels_
    unique_labels = np.unique(labels)
    # sort the groups by their median z value and label them in that order
    all_zs = []
    for label in unique_labels:
        indices_ = (labels==label)
        subpoints_um = tmp[indices_]
        z_values = subpoints_um[:,2]
        median_z = np.median(z_values)
        all_zs.append(median_z)
    all_zs = np.array(all_zs)
    order = np.argsort(all_zs)
    new_labels = np.zeros_like(labels)
    for ind, order_ind in enumerate(order):
        inds_ = indices[labels == unique_labels[order_ind]]
        plane_labels_deduplicated[inds_] = unique_labels[ind]
#%%
results['plane_labels_deduplicated'] = plane_labels_deduplicated
np.save(results_path + 'results_points_labels_deduplicated.npy', results)
#%%



row_ind = 0
subpoints_um = all_unique_points[(row_labels_deduplicated==row_ind)]
pl.figure(figsize=(5,10))
cmap = pl.get_cmap('prism')
pl.scatter(subpoints_um[:,0],subpoints_um[:,1],c=plane_labels_deduplicated[row_labels_deduplicated==row_ind],alpha=0.5,cmap=cmap)
# pl.axis('equal')
#%%



#%%
pl.figure(figsize=(5,10))
pl.plot(subpoints_um[:,0],subpoints_um[:,1],'o',alpha=0.5)
pl.plot(unique_points[:,0],unique_points[:,1],'.')
# pl.axis('equal')


#%%

#%%
from scipy.spatial import KDTree
column_ind = 19
subpoints_um = points_um[(reverse_indices_columns==column_ind)]
pl.plot(subpoints_um[:,0]*10,subpoints_um[:,1],'.')
pl.axis('equal')
#%%

tree = KDTree(subpoints)
k = 5
dists, _ = tree.query(subpoints, k=4)  # self+neighbors
md = np.median(dists[:,1:], axis=1)
pl.plot(md,'.')


#%%

row_ind = 10
column_ind = 19
subpoints = points[(reverse_indices_rows==row_ind)&(reverse_indices_columns==column_ind)]
pl.plot(subpoints[:,0]*3,subpoints[:,1]*3,'.')

