import numpy as np, healpy as hp
import os, sys
from tqdm import tqdm
import glob, h5py
import yaml
from sklearn.neighbors import BallTree
import fitsio
import joblib
from numpy.lib.recfunctions import stack_arrays

sys.path.append('/home/dhayaa/Desktop/DECADE/Balrog_of_the_decade/')
from constants import MEDSCONF

TMP_DIR = os.environ['TMPDIR']


def match_catalogs(path, tilename, bands, config):

    
    Input_catalog = fitsio.read(os.environ['CATDESDF_PATH'], ext = 1)
    
    with np.errstate(invalid = 'ignore', divide = 'ignore'):
        mag_i = 30 - 2.5*np.log10(Input_catalog['FLUX_I'])
        hlr   = np.sqrt(Input_catalog['BDF_T']) #this is only approximate
        Mask  = ((mag_i > config['gal_kws']['mag_min']) &  (mag_i < config['gal_kws']['mag_max']) &
                 (hlr > config['gal_kws']['size_min'])  &  (hlr < config['gal_kws']['size_max']))

    Input_catalog = Input_catalog[Mask]

    '/project/chihway/dhayaa/DECADE/Balrog/v08_ProductionRun3/Input_DES1210+0043-cat.fits',
    '/project/chihway/dhayaa/DECADE/Balrog/v08_ProductionRun3/SrcExtractor_DES1003-3206_i-cat.fits',
    '/project/chihway/dhayaa/DECADE/Balrog/v08_ProductionRun3/OldSrcExtractor_DES1301-3540_i-cat.fits',
    '/project/chihway/dhayaa/DECADE/Balrog/v08_ProductionRun3/metacal_DES1451-0124.fits',

    
    #Get all paths
    mcal_path  = r'%s/metacal_%s.fits' % (path, tilename)
    Truth_path = r'%s/Input_%s-cat.fits' % (path, tilename)
    OCat_path  = [r'%s/OldSrcExtractor_%s_%s-cat.fits' % (path, tilename, band) for band in bands] #Path to original SrcExtractor
    BCat_path  = [r'%s/SrcExtractor_%s_%s-cat.fits' % (path, tilename, band) for band in bands] #Path to new SrcExtractor
    
    #Read mcal, truth, and srcext catalogs
    mcal  = fitsio.read(mcal_path, ext = 1)
    Truth = fitsio.read(Truth_path, ext = 1)
    Ocat  = [fitsio.read(i, ext = 1) for i in OCat_path]
    Bcat  = [fitsio.read(i, ext = 1) for i in BCat_path]
        
    #STEP 1: match SrcExtractor objects with injected objects. Bcat[0] is r-band
    tree = BallTree(np.vstack([Bcat[0]['DELTAWIN_J2000'], Bcat[0]['ALPHAWIN_J2000']]).T * np.pi/180, leaf_size=40, metric="haversine")
    d, j = tree.query(np.vstack([Truth['dec'], Truth['ra']]).T * np.pi/180)

    d, j = d[:, 0], j[:, 0] #convert to 1d array
    d    = d * 180/np.pi * 60*60 #convert to arcsec
    
    #Keep only ids below 0.5 arcsec
    Mask = d < 0.5
    j    = j[Mask]
    Nobj = len(Mask)
    
    
    mcal_flux     = np.zeros([len(j), 3]) + np.NaN
    mcal_flux_cov = np.zeros([len(j), 3, 3]) + np.NaN
    mcal_badfrac  = np.zeros(len(j)) + np.NaN
    
    mcal_match = np.zeros(len(j), dtype = mcal.dtype)
    
    for b_i in range(len(j)):
        
        #Pick SrcExt id and find corresponding mcal id in array
        b_num = Bcat[0]['NUMBER'][j[b_i]]
        m_ind = np.where(b_num == mcal['id'])[0]
        
        #If we find a match then we take the quantities we need.
        #Also possible to not find a match (eg, if object doesnt have
        #riz band coverage).
        if len(m_ind) == 1:
            
            mcal_flux[b_i]     = mcal['mcal_flux_noshear'][m_ind]
            mcal_flux_cov[b_i] = mcal['mcal_flux_cov_noshear'][m_ind]
            mcal_badfrac[b_i]  = mcal['badfrac'][m_ind]
            
            mcal_match[b_i] = mcal[m_ind]
            
        else:
            continue
            #print("NO MATCH AT IND = ", j[b_i])
            
    
    #STEP 2: Take old, original SrcExtractor, for each truth object ask how close a nearby object is.
    tree   = BallTree(np.vstack([Ocat[0]['DELTAWIN_J2000'], Ocat[0]['ALPHAWIN_J2000']]).T * np.pi/180, leaf_size=40, metric="haversine")
    d2, j2 = tree.query(np.vstack([Truth['dec'], Truth['ra']]).T * np.pi/180)

    d2, j2 = d2[:, 0], j2[:, 0] #convert to 1d array
    d2     = d2 * 180/np.pi * 60*60 #convert to arcsec
    
    
    #STEP 3: Construct the catalog
        
    #declare type of the output array
    dtype  = np.dtype([('ID', 'i8'),  ('Truth_ind','>u4'), ('inj_class', 'i4'), ('Z', '>f4'), ('Z_SOURCE', '>i2'),
                       ('ra', '>f4'), ('dec', '>f4'), ('true_ra', '>f4'), ('true_dec', '>f4'), 
                       ('true_FLUX_r','>f4'),     ('true_FLUX_i','>f4'),     ('true_FLUX_z','>f4'), 
                       ('mcal_FLUX_r','>f4'),     ('mcal_FLUX_i','>f4'),     ('mcal_FLUX_z','>f4'), 
                       ('mcal_FLUX_r_ERR','>f4'), ('mcal_FLUX_i_ERR','>f4'), ('mcal_FLUX_z_ERR','>f4'),
                       ('IMAFLAGS_r','>f4'), ('IMAFLAGS_i','>f4'), ('IMAFLAGS_z','>f4'),
                       ('Ar','>f4'), ('Ai','>f4'), ('Az','>f4'),
                       ('d_arcsec','>f4'), ('detected', 'i4'), ('d_contam_arcsec', '>f4')])
    
    mcal_dtype = []
    
    for m_i in mcal_match.dtype.descr:
        if m_i[0] in ['expnum', 'ccdnum', 'x_exp', 'y_exp']: continue
        elif 'pars' in m_i[0]: continue
        else:
            mcal_dtype.append(m_i)
            
    mcal_dtype = np.dtype(mcal_dtype)
    
    dtype = np.dtype(dtype.descr + mcal_dtype.descr)
    output = np.zeros(Nobj, dtype = dtype)
    
    assert np.all(Truth['ID'] == Input_catalog['ID'][Truth['ind']]), "Something went wrong in matching"
    
    output['ID'] = Truth['ID']    
    
    output['inj_class'] = Truth['inj_class']
    
    output['Truth_ind'] = Truth['ind']
    output['true_ra']   = Truth['ra']
    output['true_dec']  = Truth['dec']
    output['d_arcsec']  = d
    
    output['detected']  = Mask.astype(int)
    
    output['d_contam_arcsec'] = d2
    
    output['true_FLUX_r'] = Input_catalog['FLUX_R'][Truth['ind']]
    output['true_FLUX_i'] = Input_catalog['FLUX_I'][Truth['ind']]
    output['true_FLUX_z'] = Input_catalog['FLUX_Z'][Truth['ind']]
    
    output['Z']        = Input_catalog['Z'][Truth['ind']]
    output['Z_SOURCE'] = Input_catalog['Z_SOURCE'][Truth['ind']]
    
    output['ra'][Mask]  = Bcat[0]['ALPHAWIN_J2000'][j]
    output['dec'][Mask] = Bcat[0]['DELTAWIN_J2000'][j]
    
    output['mcal_FLUX_r'][Mask] = mcal_flux[:, 0] #mcal['mcal_flux_noshear'][mcal_inds, 0]
    output['mcal_FLUX_i'][Mask] = mcal_flux[:, 1] #mcal['mcal_flux_noshear'][mcal_inds, 1]
    output['mcal_FLUX_z'][Mask] = mcal_flux[:, 2] #mcal['mcal_flux_noshear'][mcal_inds, 2]
    
    output['mcal_FLUX_r_ERR'][Mask] = np.sqrt(mcal_flux_cov[:, 0, 0])
    output['mcal_FLUX_i_ERR'][Mask] = np.sqrt(mcal_flux_cov[:, 1, 1])
    output['mcal_FLUX_z_ERR'][Mask] = np.sqrt(mcal_flux_cov[:, 2, 2])
        
    output['IMAFLAGS_r'][Mask] = Bcat[0]['FLAGS'][j]
    output['IMAFLAGS_i'][Mask] = Bcat[1]['FLAGS'][j]
    output['IMAFLAGS_z'][Mask] = Bcat[2]['FLAGS'][j]
    
    #Write non-detection rows with NaNs
    output['ra'][np.invert(Mask)]  = np.NaN
    output['dec'][np.invert(Mask)] = np.NaN
    
    output['mcal_FLUX_r'][np.invert(Mask)] = np.NaN
    output['mcal_FLUX_i'][np.invert(Mask)] = np.NaN
    output['mcal_FLUX_z'][np.invert(Mask)] = np.NaN
    
    output['mcal_FLUX_r_ERR'][np.invert(Mask)] = np.NaN
    output['mcal_FLUX_i_ERR'][np.invert(Mask)] = np.NaN
    output['mcal_FLUX_z_ERR'][np.invert(Mask)] = np.NaN
    
    output['badfrac'][np.invert(Mask)] = np.NaN
    
    for n in mcal_dtype.names:
        output[n][Mask]            = mcal_match[n]
        output[n][np.invert(Mask)] = -99
        
#     print(output[:5])
#     print(output.dtype)
#     output = np.concatenate([output, mcal_match], axis = 1)
    
    return output


if __name__ == "__main__":
    

    name     = os.path.basename(os.path.dirname(__file__))
    BROG_DIR = os.environ['BALROG_DIR']
    PATH     = BROG_DIR + '/' + name
    config   = yaml.load(open(os.path.dirname(__file__) + '/config.yaml', 'r'), Loader=yaml.Loader)
    print('GETTING BALROG FILES FROM:', PATH)
    
    files = sorted(glob.glob(PATH + '/balrog*')) + sorted(glob.glob('/project2/chihway/dhayaa/DECADE/Balrog/v09_DR3_2/balrog*'))
    tiles = [f[-17:-5] for f in files] #Tilenames
    
    FINAL_CAT = [None] * len(files)
    tilenames = [None] * len(files)
    
    def my_func(i):
        f = files[i]
        tile = os.path.basename(f).split('_')[1].split('.')[0]
        cat  = match_catalogs(os.path.dirname(f), tile, 'riz', config)
        
        return i, cat, [tile] * len(cat)
        
    #print(my_func(0))
    
    jobs = [joblib.delayed(my_func)(i) for i in range(len(files))]

    with joblib.parallel_backend("loky"):
        outputs = joblib.Parallel(n_jobs = -1, verbose=10)(jobs)
        
        for o in outputs:
            FINAL_CAT[o[0]] = o[1]
            tilenames[o[0]] = o[2]
                    
    FINAL_CAT = np.concatenate(FINAL_CAT, axis = 0)
    tilenames = np.concatenate(tilenames, axis = 0)
    
    # BITMASK = hp.read_map('/project/chihway/dhayaa/DECADE/Gold_Foreground_20230607.fits')
    BITMASK = hp.read_map('/project/chihway/dhayaa/DECADE/Foreground_Masks/GOLD_Ext0.2_Star5_MCs2_DESY6.fits')
    bmask   = BITMASK[hp.ang2pix(hp.npix2nside(BITMASK.size), FINAL_CAT['true_ra'], FINAL_CAT['true_dec'], lonlat = True)]

    with h5py.File(PATH + '/BalrogOfTheDECADE_Catalog.hdf5', 'w') as f:
    
        for i in tqdm(FINAL_CAT.dtype.names, desc = 'Making HDF5'):

            f.create_dataset(i, data = FINAL_CAT[i])
        
        f.create_dataset('FLAGS_FOREGROUND', data = bmask)
            
        f.create_dataset('tilename', data = tilenames.astype('S'), dtype = h5py.special_dtype(vlen=str))
        
        
        #Deredden quantities
        for name in ['SFD98', 'Planck13']:

            if name == 'SFD98':
                EXTINCTION = hp.read_map('/project/chihway/dhayaa/DECADE/Extinction_Maps/ebv_sfd98_nside_4096_ring_equatorial.fits')
                R_SFD98    = EXTINCTION[hp.ang2pix(4096, f['true_ra'][:], f['true_dec'][:], lonlat = True)]
                Ag, Ar, Ai, Az = R_SFD98*3.186, R_SFD98*2.140, R_SFD98*1.569, R_SFD98*1.196

            elif name == 'Planck13':
                EXTINCTION = hp.read_map('/project/chihway/dhayaa/DECADE/Extinction_Maps/ebv_planck13_nside_4096_ring_equatorial.fits')
                R_PLK13    = EXTINCTION[hp.ang2pix(4096, f['true_ra'][:], f['true_dec'][:], lonlat = True)]
                Ag, Ar, Ai, Az = R_PLK13*4.085, R_PLK13*2.744, R_PLK13*2.012, R_PLK13*1.533

            #Metacal first
            for c in ['mcal_flux_1m', 'mcal_flux_1p', 'mcal_flux_2m', 'mcal_flux_2p', 'mcal_flux_err_1m', 'mcal_flux_err_1p',
                      'mcal_flux_err_2m', 'mcal_flux_err_2p', 'mcal_flux_err_noshear', 'mcal_flux_noshear']:

                print(c + '_dered')
                arr = f[c][:]

                arr[:, 0] *= 10**(Ar/2.5)
                arr[:, 1] *= 10**(Ai/2.5)
                arr[:, 2] *= 10**(Az/2.5)

                f.create_dataset(c + '_dered_' + name.lower(), data = arr)
                
                
            f.create_dataset('mcal_FLUX_r_dered_' + name.lower(), data = f['mcal_FLUX_r'][:] * 10**(Ar/2.5))
            f.create_dataset('mcal_FLUX_i_dered_' + name.lower(), data = f['mcal_FLUX_i'][:] * 10**(Ai/2.5))
            f.create_dataset('mcal_FLUX_z_dered_' + name.lower(), data = f['mcal_FLUX_z'][:] * 10**(Az/2.5))
            
            f.create_dataset('mcal_FLUX_r_ERR_dered_' + name.lower(), data = f['mcal_FLUX_r_ERR'][:] * 10**(Ar/2.5))
            f.create_dataset('mcal_FLUX_i_ERR_dered_' + name.lower(), data = f['mcal_FLUX_i_ERR'][:] * 10**(Ai/2.5))
            f.create_dataset('mcal_FLUX_z_ERR_dered_' + name.lower(), data = f['mcal_FLUX_z_ERR'][:] * 10**(Az/2.5))
            

            f.create_dataset('Ag_' + name.lower(), data = Ag)
            f.create_dataset('Ar_' + name.lower(), data = Ar)
            f.create_dataset('Ai_' + name.lower(), data = Ai)
            f.create_dataset('Az_' + name.lower(), data = Az)
        
    
