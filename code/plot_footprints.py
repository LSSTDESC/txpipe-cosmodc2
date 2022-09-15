import cartosky
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

path_dc2 = '/global/cscratch1/sd/jprat/TXPipe/data/star-challenge/outputs/Sep14/binned_lens_catalog.hdf5'
path_gs = '/global/cscratch1/sd/jprat/TXPipe/data/gaussian_sims/outputs_gaussian_sims/gaussian_sims_srdnzs_fullsky/071222/12300area/2022/July14/binned_lens_catalog.hdf5'

colors = [ "#FADA77", "#6992C2", "#FEB580", 'tomato', 'firebrick']
labels = [  'Gaussian Simulation SRD','CosmoDC2']

dc2 = h5py.File(path_dc2, 'r')
ra_dc2 = dc2['lens/bin_0/ra'][:]
dec_dc2 = dc2['lens/bin_0/dec'][:]


gs = h5py.File(path_gs, 'r')
ra_gs = gs['lens/bin_0/ra'][:]
dec_gs = gs['lens/bin_0/dec'][:]
print('Loaded gs')
r = np.random.uniform(0,1, len(ra_gs))
mask = r<0.001

fig = plt.figure(figsize = (10,8))
smap = cartosky.Skymap(projection='ortho',lon_0=0, lat_0=-25)
smap.scatter(ra_gs[mask], dec_gs[mask], s=1.5, label= labels[1], color=colors[1])
smap.scatter(ra_dc2, dec_dc2, s=1.5, label= labels[0], color=colors[0])
legend_elements = [Line2D([0], [0], marker='o', markerfacecolor=colors[0], color='w',label=labels[0],markersize=8),
                   Line2D([0], [0], marker='o', markerfacecolor=colors[1], color='w',label=labels[1],markersize=8)]#,
                   #Line2D([0], [0], marker='o', markerfacecolor=colors[2], color='w',label=labels[2],markersize=8)]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "12"
plt.legend(loc = 'upper right', fontsize = 20,handles=legend_elements)
fig.set_size_inches([10,20])
fig.savefig('masks.png',bbox_inches="tight", dpi = 100, transparent=True, pad_inches=0.05)
fig.savefig('masks.pdf',bbox_inches="tight", transparent=True, pad_inches=0.05)

