from astropy.io import fits
import h5py
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import treecorr 
import os


############################
###        paths           #
############################
vname = '022422'
folder = '/global/projecta/projectdirs/lsst/groups/WL/users/jprat/gaussian_sims_srdnzs_fullsky/%s/'%vname
folder_cut = '/global/projecta/projectdirs/lsst/groups/WL/users/jprat/gaussian_sims_srdnzs_fullsky/%s/12300area/'%vname
filepath_txpipe_inputs = folder_cut + 'TXPipe_inputs/'

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

make_directory(filepath_txpipe_inputs)

############################
### Get ra, dec value cuts #
############################
def deg2_to_rad2(deg2):
    return deg2*(np.pi/180)**2

def h_func(A, r):
    return A/(2*np.pi) # A = 2*pi*h

def dec_lim_func(h, r):
    return np.pi/2-np.arccos(h/r)

# We want to get an area of 12,300 deg^2 (LSST Y1)

A = deg2_to_rad2(12300)
h = h_func(A, r=1)
dec_lim = -dec_lim_func(h, r=1)*180/np.pi # negative because lsst is in the south
print(h)
print(dec_lim)


############################
###   Cut sources          #
############################
for zbin in range(1,6):
    filename = 'shearcat_shearalm_zbin%d_ra_dec_g1_g2_v%s'%(zbin, vname)
    cat = np.load(folder + filename + '.npy')
    ra, dec, g1, g2 = cat.T
    mask = (dec>dec_lim)&(dec<0)
    print(len(ra))
    ra_cut, dec_cut, g1_cut, g2_cut = ra[mask], dec[mask], g1[mask], g2[mask]
    print(len(ra_cut))
    print(np.std(g1_cut))
    np.save(folder_cut  + filename + '_areacut', (ra_cut,dec_cut,g1_cut,g2_cut))


############################
###   Cut lenses           #
############################

for zbin in range(1,6):
    filename = 'galcat_%d'%zbin
    lenscat = np.load(folder + filename + '_v%s.npy'%vname)
    ra, dec = lenscat.T
    mask = (dec>dec_lim)&(dec<0)
    print(len(ra))
    ra_cut, dec_cut = ra[mask], dec[mask]
    print(len(ra_cut))
    np.save(folder_cut + filename + '_areacut', (ra_cut,dec_cut))


############################
# Now prepare TXPipe files #
############################
 
# Columns TXPipe needs from the cosmo simulation
cols = ['mag_true_u_lsst', 'mag_true_g_lsst', 
                'mag_true_r_lsst', 'mag_true_i_lsst', 
                'mag_true_z_lsst', 'mag_true_y_lsst',
                'ra', 'dec',
                'ellipticity_1_true', 'ellipticity_2_true',
                'shear_1', 'shear_2',
                'size_true',
                'galaxy_id',
                'redshift_true',
                ]


#assign a fake redshift within z lims so that TXPipe assigns it properly when source selector splits in bins
source_zbin_edges = np.array([0.19285902, 0.40831394, 0.65503818, 0.94499109, 1.2947086, 1.72779632, 2.27855242, 3. ]) # 7 bins
z_fake =[0.3, 0.5, 0.8, 1., 1.4, 1.9, 2.5]
print(z_fake)

ras, decs, g1s, g2s, zs, e1s, e2s = [], [], [], [], [], [], []

for zbin in range(1,6):
    filename = 'shearcat_shearalm_zbin%d_ra_dec_g1_g2_v%s'%(zbin, vname)
    cat = np.load(folder_cut + filename + '_areacut'+'.npy')
    ra, dec, g1, g2 = cat
    ras.extend(ra)
    decs.extend(dec)
    g1s.extend(g1)   
    g2s.extend(g2)
    zs.extend([z_fake[zbin-1]]*len(ra))
    
nobj = len(ras)

# Now fake the rest of the columns we need matching the lenth   
m_u = np.random.uniform(low=20, high=30, size=(nobj,))
m_g = np.random.uniform(low=20, high=30, size=(nobj,))
m_r = np.random.uniform(low=20, high=30, size=(nobj,))
m_i = np.random.uniform(low=20, high=30, size=(nobj,))
m_z = np.random.uniform(low=20, high=30, size=(nobj,))
m_y = np.random.uniform(low=20, high=30, size=(nobj,))
etrue1 = np.random.uniform(low=0, high=0.5, size=(nobj,))
etrue2 = np.random.uniform(low=0, high=0.5, size=(nobj,))
size = np.random.uniform(low=2., high=10., size=(nobj,))
galaxy_id = np.arange(0,nobj, 1, dtype=int)

ras, decs, g1s, g2s, zs = np.array(ras), np.array(decs), np.array(g1s), np.array(g2s), np.array(zs) 
filepath_save = folder_cut + 'shearcat_shearalm_allbins_ra_dec_g1_g2_and_fakecols' + '_areacut'
print('Paste this filename to config file in InputCats stage:', filepath_save)
np.save(filepath_save, (ras,decs,g1s,g2s,zs,m_u,m_g, m_r, m_i, m_z, m_y, etrue1, etrue2, size, galaxy_id))

################################
# Produce the lens_catalog.hdf5#
################################
ras, decs = [], []

for zbin in range(1,6):
    cat = np.load(folder_cut + 'galcat_%d'%zbin + '_areacut.npy')
    ra, dec = cat
    ras.extend(ra)
    decs.extend(dec)

ras, decs = np.array(ras), np.array(decs)
filepath_save = filepath_txpipe_inputs + 'lens_catalog'

# need to save this in hdf5 format
hf = h5py.File('%s.hdf5'%filepath_save, 'w')
g_lens = hf.create_group('lens')
g_lens.create_dataset('ra',data=ras)
g_lens.create_dataset('dec',data=decs)
hf.close()

print('Paste this filename to pipeline file:', filepath_save)

# Produce the lens_tomography_catalog.hdf5
zbins, counts = [], []

for zbin in range(1,6):
    cat = np.load(folder_cut + 'galcat_%d'%zbin + '_areacut.npy')
    ra, dec = cat
    zbins.extend([zbin-1]*len(ra)) #starts with 0 in txpipe
    counts.append(len(ra))

zbins, counts = np.array(zbins), np.array(counts)
counts_2d = np.array([counts.sum()])
ws = np.array([1.]*counts_2d[0])

filepath_save = filepath_txpipe_inputs + 'lens_tomography_catalog'

# need to save this in hdf5 format
hf = h5py.File('%s.hdf5'%filepath_save, 'w')
t_lens = hf.create_group('tomography')
t_lens.create_dataset('lens_bin',data=zbins)
t_lens.create_dataset('lens_counts',data=counts)
t_lens.create_dataset('lens_counts_2d',data=counts_2d )
t_lens.create_dataset('lens_weight',data=ws)
t_lens.attrs['nbin_lens'] = 5
hf.close()
print('Paste this filename to pipeline file:', filepath_save)


#############################
# Redshift nzs files        #
#############################

# Load SRD nzs
lens = np.loadtxt('../generation_gaussian_sims/srd_nzs/nz_y1_lens_5bins_srd.txt').T
source = np.loadtxt('../generation_gaussian_sims/srd_nzs/nz_y1_srcs_5bins_srd.txt').T

z_l = lens[0]
z_s = source[0]
nzs_l = lens[1:]
nzs_s = source[1:]

# Lens sample
pz_l = h5py.File(filepath_txpipe_inputs + 'lens_photoz_stack.hdf5', 'w')
n_of_z = pz_l.create_group('n_of_z')
lens = pz_l.create_group('n_of_z/lens')

l_bins = 5
lens.create_dataset('z',data=z_l)
for i in range(l_bins):
    lens.create_dataset('bin_%d'%i,data=nzs_l[i])
lens.attrs['nbin'] = l_bins
pz_l.close()


# source sample
pz_s = h5py.File(filepath_txpipe_inputs + 'shear_photoz_stack.hdf5', 'w')
n_of_z = pz_s.create_group('n_of_z')
source = pz_s.create_group('n_of_z/source')
source2d = pz_s.create_group('n_of_z/source2d')

source.create_dataset('z',data=z_s)
source2d.create_dataset('z',data=z_s)

s_bins = 5
for i in range(s_bins):
    source.create_dataset('bin_%d'%i,data=nzs_s[i])
source2d.create_dataset('bin_0',data=nzs_s)
source.attrs['nbin'] = s_bins
pz_s.close()


print('Paste this filename to pipeline file:', filepath_txpipe_inputs + 'lens_photoz_stack.hdf5')
print('Paste this filename to pipeline file:', filepath_txpipe_inputs + 'shear_photoz_stack.hdf5')


#############################
#      mask                 #
#############################


lens_cat_check = h5py.File(filepath_txpipe_inputs + 'lens_catalog.hdf5', 'r')
ra = lens_cat_check['lens/ra'][:]
dec = lens_cat_check['lens/dec'][:]


phi = ra*np.pi/180. 
theta = np.pi/2 - dec*np.pi/180.
nside = 2048
pix = hp.ang2pix(nside,theta,phi)
map=  np.bincount(pix,minlength=hp.nside2npix(nside))
map[map>1]=1

filled_pixels = np.unique(pix)
value_of_filled_pixels = np.ones(len(filled_pixels))

# Lens sample
mask_file = h5py.File(filepath_txpipe_inputs + 'mask.hdf5', 'w')
maps_group = mask_file.create_group('maps')
mask_group = mask_file.create_group('maps/mask')

mask_group.create_dataset('pixel',data=filled_pixels)
mask_group.create_dataset('value',data=value_of_filled_pixels)
mask_group.attrs['area'] = 12300
mask_group.attrs['bright_obj_threshold'] = 22.
mask_group.attrs['chunk_rows'] = 100000
mask_group.attrs['depth_band'] = 'i'
mask_group.attrs['f_sky'] = 12300/41252.96125
mask_group.attrs['nest'] = False
mask_group.attrs['npix'] = hp.nside2npix(2048)
mask_group.attrs['nside'] = 2048
mask_group.attrs['pixelization'] = 'healpix'
mask_group.attrs['snr_delta'] = 10
mask_group.attrs['snr_threshold'] = 10
mask_group.attrs['sparse'] = True
mask_file.close()
print('Paste this filename to pipeline file:', filepath_txpipe_inputs + 'mask.hdf5')
