import numpy as np
import os, sys
from tqdm import tqdm
import glob, h5py
import yaml
from sklearn.neighbors import BallTree
import fitsio

from files import get_mcal_file_path, get_truth_catalog_path, get_band_info_file, get_balrog_file_path
from constants import MEDSCONF

TMP_DIR = os.environ['TMPDIR']

def match_catalogs(tilename, bands, output_desdata, config):

    Input_catalog = fitsio.read(os.environ['CATDESDF_PATH'], ext = 1)
    
    mag_i = 30 - 2.5*np.log10(Input_catalog['FLUX_I'])
    hlr   = np.sqrt(Input_catalog['BDF_T']) #this is only approximate
    Mask  = ((mag_i > config['gal_kws']['mag_min']) &  (mag_i < config['gal_kws']['mag_max']) &
             (hlr > config['gal_kws']['size_min'])  &  (hlr < config['gal_kws']['size_max']))

    print(np.average(Mask), np.sum(Mask))

    Input_catalog = Input_catalog[Mask]

    #Get all paths
    mcal_path  = get_mcal_file_path(meds_dir=output_desdata, medsconf=MEDSCONF, tilename=tilename)
    brog_path  = get_balrog_file_path(meds_dir=output_desdata, medsconf=MEDSCONF, tilename=tilename)
    Truth_path = get_truth_catalog_path(meds_dir=output_desdata, medsconf=MEDSCONF, tilename=tilename)
    Binfo_path = {b : get_band_info_file(meds_dir=output_desdata, medsconf=MEDSCONF, tilename=tilename, band = b) for b in bands}
    info       = {band : yaml.load(open(Binfo_path[band], 'r'), Loader=yaml.Loader) for band in bands}
    OCat_path  = [info[band]['cat_path'] for band in bands] #Path to original SrcExtractor
    BCat_path  = [info[band]['cat_path'].replace(TMP_DIR, output_desdata) for band in bands] #Path to new SrcExtractor
    
    #Read mcal, truth, and srcext catalogs
    mcal  = fitsio.read(mcal_path, ext = 1)
    Truth = fitsio.read(Truth_path, ext = 1)
    Ocat  = [fitsio.read(i, ext = 1) for i in OCat_path]
    Bcat  = [fitsio.read(i, ext = 1) for i in BCat_path]
    
    SrcExt_r = Bcat[0] #use just r-band for getting position + ID (This is same in all bands). This needs to be new SrcExt cat
    Mcal_ID  = mcal['id']
    RA, DEC  = SrcExt_r['ALPHAWIN_J2000'], SrcExt_r['DELTAWIN_J2000']
    number   = SrcExt_r['NUMBER']
    
    #Match metacal with source extractor (needed to get ra/dec)
    inds = np.intersect1d(number, Mcal_ID, return_indices = True)[1]
    
    #STEP 1: match SrcExtractor objects with injected objects
    tree = BallTree(np.vstack([DEC[inds], RA[inds]]).T * np.pi/180, leaf_size=2, metric="haversine")
    d, j = tree.query(np.vstack([Truth['dec'], Truth['ra']]).T * np.pi/180)

    d, j = d[:, 0], j[:, 0] #convert to 1d array
    d    = d * 180/np.pi * 60*60 #convert to arcsec
    
    #Keep only ids below 0.5 arcsec
    Mask = d < 0.5
    j    = j[Mask]
    Nobj = len(Mask)
    
    
    
    #STEP 2: Take old, original SrcExtractor, for each truth object ask how close a nearby object is.
    tree = BallTree(np.vstack([Ocat[0]['DELTAWIN_J2000'], Ocat[0]['ALPHAWIN_J2000']]).T * np.pi/180, leaf_size=2, metric="haversine")
    d2, j2 = tree.query(np.vstack([Truth['dec'], Truth['ra']]).T * np.pi/180)

    d2, j2 = d2[:, 0], j2[:, 0] #convert to 1d array
    d2     = d2 * 180/np.pi * 60*60 #convert to arcsec
    
    
    #STEP 3: Construct the catalog
        
    #declare type of the output array
    dtype  = np.dtype([('ID', 'i8'),  ('Truth_ind','>u4'), ('inj_class', 'i4'),
                       ('ra', '>f4'),('dec', '>f4'),('true_ra', '>f4'), ('true_dec', '>f4'), 
                       ('true_FLUX_r','>f4'),('true_FLUX_i','>f4'),('true_FLUX_z','>f4'), 
                       ('FLUX_r','>f4'),     ('FLUX_i','>f4'),     ('FLUX_z','>f4'), 
                       ('FLUX_r_ERR','>f4'), ('FLUX_i_ERR','>f4'), ('FLUX_z_ERR','>f4'),
                       ('IMAFLAGS_r','>f4'), ('IMAFLAGS_i','>f4'), ('IMAFLAGS_z','>f4'),
                       ('Ar','>f4'), ('Ai','>f4'), ('Az','>f4'),
                       ('d_arcsec','>f4'), ('detected', 'i4'), ('d_contam_arcsec', '>f4')])
    
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
    
    output['ra'][Mask]  = RA[inds][j]
    output['dec'][Mask] = DEC[inds][j]
    
    output['FLUX_r'][Mask] = mcal['mcal_flux_noshear'][j, 0]
    output['FLUX_i'][Mask] = mcal['mcal_flux_noshear'][j, 1]
    output['FLUX_z'][Mask] = mcal['mcal_flux_noshear'][j, 2]
    
    output['FLUX_r_ERR'][Mask] = np.sqrt(mcal['mcal_flux_cov_noshear'][j, 0, 0])
    output['FLUX_i_ERR'][Mask] = np.sqrt(mcal['mcal_flux_cov_noshear'][j, 1, 1])
    output['FLUX_z_ERR'][Mask] = np.sqrt(mcal['mcal_flux_cov_noshear'][j, 2, 2])
    
    output['IMAFLAGS_r'][Mask] = Bcat[0]['FLAGS'][inds][j]
    output['IMAFLAGS_i'][Mask] = Bcat[1]['FLAGS'][inds][j]
    output['IMAFLAGS_z'][Mask] = Bcat[2]['FLAGS'][inds][j]
    
    #Write non-detection rows with NaNs
    output['ra'][np.invert(Mask)]  = np.NaN
    output['dec'][np.invert(Mask)] = np.NaN
    
    output['FLUX_r'][np.invert(Mask)] = np.NaN
    output['FLUX_i'][np.invert(Mask)] = np.NaN
    output['FLUX_z'][np.invert(Mask)] = np.NaN
    
    output['FLUX_r_ERR'][np.invert(Mask)] = np.NaN
    output['FLUX_i_ERR'][np.invert(Mask)] = np.NaN
    output['FLUX_z_ERR'][np.invert(Mask)] = np.NaN
    
    fitsio.write(brog_path, output)