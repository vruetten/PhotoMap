#%%
import numpy as np
from glob import glob
import tifffile as tf
import matplotlib.pyplot as pl
import matplotlib as mpl
import sys
from mesoscope.napari import napari_utils
import skimage
#from scipy import ndimage as ndi
from importlib import reload
import warnings
from pathlib import Path
import skimage


from importlib import reload
reload(napari_utils)


def compute_gridiness(grid_yx, L=90, weights=(1.0, 1.0, 0.5)):
    ny, nx, _ = grid_yx.shape
    G = np.full((ny, nx, 3), np.nan)

    w1, w2, w3 = weights

    for i in range(ny):
        for j in range(nx):
            center = grid_yx[i, j]
            if np.any(np.isnan(center)):
                continue

            neighbors = {}
            for dy, dx, name in [(-1, 0, 'up'), (1, 0, 'down'), (0, -1, 'left'), (0, 1, 'right')]:
                ni, nj = i + dy, j + dx
                if 0 <= ni < ny and 0 <= nj < nx:
                    neighbor = grid_yx[ni, nj]
                    if not np.any(np.isnan(neighbor)):
                        neighbors[name] = neighbor

            deviations = []
            for dir in ['right', 'down']:
                if dir in neighbors:
                    v = neighbors[dir] - center
                    deviations.append(1-abs(np.linalg.norm(v)/L))
            length_dev = np.mean(deviations) if deviations else np.nan

            if 'right' in neighbors and 'down' in neighbors:
                vx = neighbors['right'] - center
                vy = neighbors['down'] - center
                vx_norm = np.linalg.norm(vx)
                vy_norm = np.linalg.norm(vy)
                if vx_norm > 0 and vy_norm > 0:
                    cos_angle = np.dot(vx, vy) / (vx_norm * vy_norm)
                    ortho_dev = cos_angle
                else:
                    ortho_dev = np.nan
            else:
                ortho_dev = np.nan

            sym_devs = []
            if 'left' in neighbors and 'right' in neighbors:
                v1 = neighbors['left'] - center
                v2 = neighbors['right'] - center
                sym_devs.append(np.linalg.norm(v1 + v2)/L)
            if 'up' in neighbors and 'down' in neighbors:
                v1 = neighbors['up'] - center
                v2 = neighbors['down'] - center
                sym_devs.append(np.linalg.norm(v1 + v2)/L)
            sym_dev = np.mean(sym_devs) if sym_devs else 0

            distortion = w1 * length_dev + w2 * ortho_dev + w3 * sym_dev
            G[i,j, 0] = length_dev
            G[i,j, 1] = ortho_dev
            G[i,j, 2] = sym_dev
            # G[i, j] = 1 - min(distortion, 1)

    return G
#%%

# === CONFIGURATION ===
image_path =  '/nrs/ahrens/Virginia_nrs/wbi_paper_data/photomap/250730_exm_photomap/post/sagittal/composite_image_250730_sagittal_post_db_00001_TZCXY_ds3.tif'     # 3D image, shape (Z, Y, X)

save_path   = '/nrs/ahrens/Virginia_nrs/wbi_paper_data/photomap/250730_exm_photomap/post/sagittal/points/'
# === LOAD DATA ===
# image = np.load(image_path)

########################## LOAD RESULTS
image_path2 = '/nrs/ahrens/Virginia_nrs/wbi_paper_data/photomap/250730_exm_photomap/post/sagittal/points/planes2.tif'
data = tf.imread(image_path2)
#%%
grid_results_path = '/nrs/ahrens/Virginia_nrs/wbi_paper_data/photomap/250730_exm_photomap/post/sagittal/points/grid_results_napari_fixed.npy'
grid_results = np.load(grid_results_path, allow_pickle=True).item()

points = grid_results['napari_points']
row_labels = grid_results['napari_row_labels'].astype(int)
column_labels = grid_results['napari_column_labels'].astype(int)
print(f"number of unique row labels: {len(np.unique(row_labels))}")
print(f"number of unique column labels: {len(np.unique(column_labels))}")
fish_labels = grid_results['napari_fish_labels'].astype(int)
grid_points = grid_results['grid_points']
grid_labels = grid_results['grid_labels']
bad_columns = [15]
good_columns = [c for c in range(grid_points.shape[2]) if c not in bad_columns]
grid_points = grid_points[:,:,good_columns]
grid_labels = grid_labels[:,:, good_columns]
# fish_labels = fish_labels[:,:,good_columns]

#%%
L = 90
weights = (1.0, 0.0, 0.0)
weights = (0.0, 1.0, 0.0)
weights = (0.0, 0.0, 1.0)
weights = (1.0, 1.0, 0.5)
weights = weights/np.sum(weights)
gridiness = np.stack([compute_gridiness(grid_yx, L=L, weights=weights) for grid_yx in grid_points], axis=0)
gridiness_on_fish = gridiness[grid_labels ==0]
gridiness_off_fish = gridiness[grid_labels >0]


length_dev = gridiness[:,:,:,0]
length_dev_on_fish = length_dev[grid_labels ==1]
length_dev_off_fish = length_dev[grid_labels ==0]
ortho_dev = gridiness[:,:,:,1]
ortho_dev_on_fish = ortho_dev[grid_labels ==1]
ortho_dev_off_fish = ortho_dev[grid_labels ==0]
sym_dev = gridiness[:,:,:,2]
sym_dev_on_fish = sym_dev[grid_labels ==1]
sym_dev_off_fish = sym_dev[grid_labels ==0]
#%%
from scipy.stats import gaussian_kde
density = True
cumulative = False
stacked = False
pl.figure(figsize=(2,1.5))
pl.title(f"length deviation")
tmp0 = length_dev_on_fish.flatten()
tmp1 = length_dev_off_fish.flatten()
tmp0 = tmp0[~np.isnan(tmp0)]
tmp1 = tmp1[~np.isnan(tmp1)]
# Define KDE functions
kde0 = gaussian_kde(tmp0)
kde1 = gaussian_kde(tmp1)
# Define common evaluation grid
x_eval = np.linspace(-0.5, 0.5, 500)

# Evaluate KDEs
pdf0 = kde0(x_eval)
pdf1 = kde1(x_eval)

# Calculate common bin edges based on combined data range
combined_data = np.concatenate([tmp0, tmp1])
bin_edges = np.linspace(combined_data.min(), combined_data.max(), 101)  # 51 edges for 50 bins
# pl.hist(tmp0, bins = bin_edges, alpha = 1, label = 'on fish', stacked = stacked, color = pl.get_cmap('tab10')(1), zorder = -1, density = density, cumulative = cumulative)
pl.plot(x_eval, pdf1, label='off fish', color=pl.get_cmap('tab10')(0), lw=2, alpha=1, zorder = -1)
pl.fill_between(x_eval, pdf1, 0, color=pl.get_cmap('tab10')(0), alpha=0.8, zorder = -1)


pl.plot(x_eval, pdf0, label='on fish', color=pl.get_cmap('tab10')(1), lw=2, alpha=0.7)
pl.fill_between(x_eval, pdf0, 0, color=pl.get_cmap('tab10')(1), alpha=0.7)
# pl.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# pl.hist(tmp1, bins = bin_edges, alpha = 0.7, label = 'off fish', stacked = stacked, color = pl.get_cmap('tab10')(0), density = density, cumulative = cumulative)

# pl.legend(bbox_to_anchor=(1.05, 0.4), loc='upper left')
pl.xlabel('length deviation')
pl.ylabel('density')
pl.xlim([-0.5, 0.5])
pl.ylim(0,27)
pl.savefig(save_path + 'length_deviation_kde_not_flipped.pdf')
# pl.ylabel('cumulative density')
#%%
from scipy.stats import gaussian_kde
density = True
cumulative = False
stacked = False
pl.figure(figsize=(2,1.5))
pl.title(f"angle deviation")
tmp0 = ortho_dev_on_fish.flatten()
tmp1 = ortho_dev_off_fish.flatten()
tmp0 = tmp0[~np.isnan(tmp0)]
tmp1 = tmp1[~np.isnan(tmp1)]
# Define KDE functions
kde0 = gaussian_kde(tmp0)
kde1 = gaussian_kde(tmp1)
# Define common evaluation grid
x_eval = np.linspace(-0.5, 0.5, 500)

# Evaluate KDEs
pdf0 = kde0(x_eval)
pdf1 = kde1(x_eval)

# Calculate common bin edges based on combined data range
combined_data = np.concatenate([tmp0, tmp1])
bin_edges = np.linspace(combined_data.min(), combined_data.max(), 101)  # 51 edges for 50 bins
# pl.hist(tmp0, bins = bin_edges, alpha = 1, label = 'on fish', stacked = stacked, color = pl.get_cmap('tab10')(1), zorder = -1, density = density, cumulative = cumulative)
pl.plot(x_eval, pdf1, label='off fish', color=pl.get_cmap('tab10')(0), lw=2, alpha=1, zorder = -1)
pl.fill_between(x_eval, pdf1, 0, color=pl.get_cmap('tab10')(0), alpha=0.8, zorder = -1)

pl.plot(x_eval, pdf0, label='on fish', color=pl.get_cmap('tab10')(1), lw=2, alpha=0.7)
pl.fill_between(x_eval, pdf0, 0, color=pl.get_cmap('tab10')(1), alpha=0.7)
# pl.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# pl.hist(tmp1, bins = bin_edges, alpha = 0.7, label = 'off fish', stacked = stacked, color = pl.get_cmap('tab10')(0), density = density, cumulative = cumulative)

# pl.legend(bbox_to_anchor=(1.05, 0.4), loc='upper left')
pl.xlabel('angle deviation')
pl.ylabel('density')
pl.xlim([-0.5, 0.5])
pl.ylim(0,27)
pl.savefig(save_path + 'angle_deviation_kde_not_flipped.pdf')
#%%


#%%
from matplotlib import colors as mcolors
# for plane in range(data.shape[0]):
for plane in [6]:
    # subdata = data[plane,:,:]
    # subdata = subdata - subdata.min()
    # subdata_gamma = skimage.exposure.adjust_gamma(subdata, 1.2)

    subpoints = grid_points[plane][:,:,1:].reshape(-1,2)
    sublabels = np.abs(gridiness[plane][:,:,0].reshape(-1))
    
    vmin = 0
    vmax = 0.5
    vcenter = 0.25
    vnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # Get colormap and mapped colors
    cmap = pl.get_cmap('magma_r')
    colors = cmap(vnorm(sublabels))  # mapped RGBA colors


    pl.figure(figsize=(8,16))
    pl.imshow(subdata, cmap='gray')
    pl.clim([40, 200])
    sc = pl.scatter(subpoints[:,1], subpoints[:,0], c=sublabels, s=20, norm=vnorm, cmap=cmap)
    pl.colorbar(sc, label='length deviation', shrink=0.2)  # optional: add colorbar
    

    # sublabels = grid_labels[plane].reshape(-1).astype(int)
    # colors = pl.get_cmap('tab10')([1,0])
    # colors = colors[sublabels]
    # pl.scatter(subpoints[:,1], subpoints[:,0], c=colors, s=20)
    num_rows = grid_points.shape[1]
    num_columns = grid_points.shape[2]
    subpoints = grid_points[plane][:,:,:]
    for ind,row in enumerate(range(num_rows)):
        color = pl.get_cmap('magma')(ind/num_rows)
        pl.plot(subpoints[row,:,2],subpoints[row,:,1],'-',alpha=1, color = 'k', zorder = -1, lw = 1)
    for ind,column in enumerate(range(num_columns)):

        pl.plot(subpoints[:,column,2],subpoints[:,column,1],'-',alpha=1, color = 'k', zorder = -1, lw = 1)

    pl.axis('equal')
    pl.axis('off')
    pl.savefig(save_path + f'plane_{plane}_with_points_large_griddiness_p3_dual_final_length.pdf')
 #%%
from matplotlib import colors as mcolors
# for plane in range(data.shape[0]):
for plane in [6]:
    # subdata = data[plane,:,:]
    # subdata = subdata - subdata.min()
    # subdata_gamma = skimage.exposure.adjust_gamma(subdata, 1.2)

    subpoints = grid_points[plane][:,:,1:].reshape(-1,2)
    sublabels = np.abs(gridiness[plane][:,:,1].reshape(-1))
    
    vmin = 0
    vmax = 0.5
    vcenter = 0.25
    vnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # Get colormap and mapped colors
    cmap = pl.get_cmap('magma_r')
    colors = cmap(vnorm(sublabels))  # mapped RGBA colors


    pl.figure(figsize=(8,16))
    pl.imshow(subdata, cmap='gray')
    pl.clim([40, 200])
    sc = pl.scatter(subpoints[:,1], subpoints[:,0], c=sublabels, s=20, norm=vnorm, cmap=cmap)
    pl.colorbar(sc, label='angle deviation', shrink=0.2)  # optional: add colorbar
    

    # sublabels = grid_labels[plane].reshape(-1).astype(int)
    # colors = pl.get_cmap('tab10')([1,0])
    # colors = colors[sublabels]
    # pl.scatter(subpoints[:,1], subpoints[:,0], c=colors, s=20)
    num_rows = grid_points.shape[1]
    num_columns = grid_points.shape[2]
    subpoints = grid_points[plane][:,:,:]
    for ind,row in enumerate(range(num_rows)):
        color = pl.get_cmap('magma')(ind/num_rows)
        pl.plot(subpoints[row,:,2],subpoints[row,:,1],'-',alpha=1, color = 'k', zorder = -1, lw = 1)
    for ind,column in enumerate(range(num_columns)):

        pl.plot(subpoints[:,column,2],subpoints[:,column,1],'-',alpha=1, color = 'k', zorder = -1, lw = 1)

    pl.axis('equal')
    pl.axis('off')
    pl.savefig(save_path + f'plane_{plane}_with_points_large_griddiness_p3_dual_final_angle.pdf')   
# %%
num_rows = grid_points.shape[1]
num_columns = grid_points.shape[2]
subpoints = grid_points[6][:,:,:]
pl.figure(figsize=(5,3))
for ind,row in enumerate(range(num_rows)):
    color = pl.get_cmap('magma')(ind/num_rows)
    pl.plot(subpoints[row,:,1],subpoints[row,:,2],'o-',alpha=1, color = color)
pl.figure(figsize=(5,3))
for ind,column in enumerate(range(num_columns)):
    color = pl.get_cmap('magma')(ind/num_columns)
    pl.plot(subpoints[:,column,1],subpoints[:,column,2],'o-',alpha=1, color = color)
#%%
plane = 6
subdata = data[plane,:,:]
subdata = subdata - subdata.min()
subdata_gamma = skimage.exposure.adjust_gamma(subdata, 1.2)
num_rows = grid_points.shape[1]
num_columns = grid_points.shape[2]
subpoints = grid_points[plane][:,:,:]
pl.figure(figsize=(8,16))
pl.imshow(subdata_gamma, cmap='gray')
# pl.clim([40, 200])
for ind,row in enumerate(range(num_rows)):
    color = pl.get_cmap('prism')(ind/num_rows)
    pl.plot(subpoints[row,:,2],subpoints[row,:,1],'.-',alpha=1, color = color)

bad_columns = [26, 28, 31]
good_columns = [c for c in range(num_rows) if c not in bad_columns]
for ind,column in enumerate(range(num_columns)):
        color = pl.get_cmap('prism')(ind/num_columns)
        pl.plot(subpoints[good_columns,column,2],subpoints[good_columns,column,1],'.-',alpha=1, color = color)
pl.axis('equal')
pl.axis('off')
# pl.savefig(save_path + f'plane_{plane}_with_points_large_griddiness_p3_dual_rows_cols.pdf')
#%%
