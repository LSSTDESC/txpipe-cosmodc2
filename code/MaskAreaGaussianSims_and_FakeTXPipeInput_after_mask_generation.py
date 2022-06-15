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
vname = '051422'
folder = '/global/cfs/cdirs/lsst/groups/WL/users/jprat/gaussian_sims_srdnzs_fullsky/%s/'%vname
folder_cut = '/global/cfs/cdirs/lsst/groups/WL/users/jprat/gaussian_sims_srdnzs_fullsky/%s/12300area/'%vname
filepath_txpipe_inputs = folder_cut + 'TXPipe_inputs/'

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

make_directory(filepath_txpipe_inputs)

# Load mask we generated in MaskAreaGaussianSims_and_FakeTXPipeInput.py to cut lenses and sources in this way
# instead of an ra cut, since that creates problems at the edge of the mask.

mask_path = folder_cut + 'TXPipe_inputs/mask.hdf5'
mask_hdf5 = h5py.File(mask_path, 'r')


name = 'mask' 
nside = 2048
npix = hp.nside2npix(nside)
mask_map = np.zeros(npix)
pix_m = mask_hdf5['maps/%s'%name]['pixel'][:]
value_m = mask_hdf5['maps/%s'%name]['value'][:]
print(value_m)
mask_map[pix_m]= value_m




def to_ra_dec(theta,phi):
    ra = phi*180./np.pi
    dec = 90. - theta*180./np.pi
    return ra, dec


def to_phi_theta(ra, dec):
    phi = ra*np.pi/180.
    theta = (90.-dec)*np.pi/180.
    return phi, theta

'''
############################
###   Cut lenses           #
############################

for zbin in range(0,5):
    filename = 'lenscat%d'%zbin
    lenscat = np.load(folder + filename + '.npy')
    ra, dec = lenscat.T
    # Cut in dec because some galaxies have unphysical values by little, due to details of sims.
    mask_bad_objects = (dec<90) & (dec>-90)
    dec = dec[mask_bad_objects]
    ra = ra[mask_bad_objects]
    #mask = (dec>dec_lim)&(dec<0)
    # cut sources with mask instead of dec cut
    phi, theta = to_phi_theta(ra, dec)
    pix_d = hp.ang2pix(nside, theta, phi)  
    mask = mask_map[pix_d]==1          
    print('Lens %d:'%zbin, len(ra))
    ra_cut, dec_cut = ra[mask], dec[mask]
    print(len(ra_cut))
    np.save(folder_cut + filename + '_areacut', (ra_cut,dec_cut))


############################
###   Cut sources          #
############################
for zbin in range(0,5):
    filename = 'shearcat%d'%(zbin)
    cat = np.load(folder + filename + '.npy')
    ra, dec, e1, e2, _, _  = cat.T # last two columns are noise
    # cut sources with mask instead of dec cut
    phi, theta = to_phi_theta(ra, dec)
    pix_d = hp.ang2pix(nside, theta, phi)  
    mask = mask_map[pix_d]==1          
    print('Source %d:'%zbin, len(ra))
    ra_cut, dec_cut, e1_cut, e2_cut = ra[mask], dec[mask], e1[mask], e2[mask]
    print(len(ra_cut))
    print('np.std(e1_cut)', np.std(e1_cut))
    np.save(folder_cut  + filename + '_areacut', (ra_cut,dec_cut,e1_cut,e2_cut))
'''

'''
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

ras, decs, e1s, e2s, zs, e1s, e2s = [], [], [], [], [], [], []

for zbin in range(0,5):
    filename = 'shearcat%d'%(zbin)
    cat = np.load(folder_cut + filename + '_areacut'+'.npy')
    ra, dec, e1, e2 = cat
    ras.extend(ra)
    decs.extend(dec)
    e1s.extend(e1)   
    e2s.extend(e2)
    zs.extend([z_fake[zbin]]*len(ra))
    
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

ras, decs, e1s, e2s, zs = np.array(ras), np.array(decs), np.array(e1s), np.array(e2s), np.array(zs) 
filepath_save = folder_cut + 'shearcat_allbins_ra_dec_e1_e2_and_fakecols' + '_areacut'
print('Paste this filename to config file in InputCats stage:', filepath_save)
np.save(filepath_save, (ras,decs,e1s,e2s,zs,m_u,m_g, m_r, m_i, m_z, m_y, etrue1, etrue2, size, galaxy_id))
'''
################################
# Produce the lens_catalog.hdf5#
################################
ras, decs = [], []

for zbin in range(0,5):
    cat = np.load(folder_cut + 'lenscat%d'%zbin + '_areacut.npy')
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

for zbin in range(0,5):
    cat = np.load(folder_cut + 'lenscat%d'%zbin + '_areacut.npy')
    ra, dec = cat
    zbins.extend([zbin]*len(ra)) #starts with 0 in txpipe
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


