import sys
sys.path.append('..')

from src.lenses import LensGalaxy
import numpy as np
import skysurvey
from src.glsne_class import GLSNe
import sncosmo
from astropy.cosmology import Planck18 as cosmo
from skysurvey.tools import utils
from skysurvey import DataSet
import pickle
import os
import pandas as pd
import sfdmap

def hostdust_Ia(r_v=2., ebv_rate=0.11, size=None):
    hostr_v = r_v * np.ones(size)
    hostebv = np.random.exponential(ebv_rate, size)
    return hostr_v, hostebv


survey = pd.read_pickle('../../LENSIT/input/logs/ztf_survey_p12_allsurvey.pkl')

from skysurvey import ztf

ztf_survey = ztf.ZTF(survey, level='quadrant')

starting_date, ending_date = 58288.171875, 60236.54296875

lens = LensGalaxy()

dust = sncosmo.CCM89Dust()


class glSNIa(skysurvey.SNeIa):
    _KIND = "GLSNIa"
    sntemplate = "salt2-extended"
    source = GLSNe(sntemplate, 4, name=_KIND)
    _TEMPLATE = sncosmo.Model(source, effects=[dust, dust, dust], effect_names=['MW', 'host', 'lens'],
                              effect_frames=['obs', 'rest', 'free'])


# -----------------------
size = 10000
file_index = 63
z_max = 1.5
ndet_total = 37545
# -------------- ---------

while ndet_total < 100000:
    
    lensfile = '../sim_output/lenses_' + str(int(size)) + '_' + str(int(file_index)) + '.pkl'
    if os.path.exists(lensfile):
        lenses = pd.read_pickle(lensfile)
    else:
        lenses = lens.sample_uniform_zs(z_min=0.1, z_max=z_max, size=size, mu_total_min=2)
        lenses['Asl'] = lens.ASL(lenses['zlens'].values, lenses['zsource'].values, lenses['sigma'].values)
        lenses.to_pickle(lensfile)
        print('Saved: ', lensfile)

    data = pd.DataFrame(lenses)

    data['z'] = data['zsource'].values
    data['t0'] = np.random.uniform(starting_date, ending_date, size=len(data))
    data['dt_1'] = data['dt_1'].values / (1. + data['z'].values)
    data['dt_2'] = data['dt_2'].values / (1. + data['z'].values)
    data['dt_3'] = data['dt_3'].values / (1. + data['z'].values)
    data['dt_4'] = data['dt_4'].values / (1. + data['z'].values)
    data['mu_1'] = data['mu_1'].values
    data['mu_2'] = data['mu_2'].values
    data['mu_3'] = data['mu_3'].values
    data['mu_4'] = data['mu_4'].values
    x, pdf = skysurvey.target.snia.SNeIaStretch.nicolas2021()
    data['x1'] = np.random.choice(x, size=len(data), p=pdf / np.sum(pdf))
    x, pdf = skysurvey.target.snia.SNeIaColor.intrinsic_and_dust()
    data['c'] = np.random.choice(x, size=len(data), p=pdf / np.sum(pdf))
    data['MB'] = skysurvey.target.snia.SNeIaMagnitude.tripp1998(data['x1'], data['c'])
    snia = skysurvey.SNeIa()
    data['MB_obs'] = snia.magabs_to_magobs(data['z'], data['MB'], cosmology=cosmo)
    data['x0'] = snia.magobs_to_amplitude(data['MB_obs'], param_name='x0')
    data['ra'], data['dec'] = utils.random_radec(ra_range=[0, 360], dec_range=[-30, 90], size=len(data))
    
    mw_sfdmap = sfdmap.SFDMap(mapdir='../../LENSIT/input/sfddata-master')
    data['MWebv'] = mw_sfdmap.ebv(data['ra'], data['dec'])
    data['hostr_v'], data['hostebv'] = hostdust_Ia(size=len(data))

    glsnia = glSNIa.from_data(data)

    dset = DataSet.from_targets_and_survey(glsnia, ztf_survey)

    del glsnia

    det = dset.get_ndetection()
    dset.targets.data['ndet'] = np.nan
    dset.targets.data['ndet'].loc[det.index] = det

    dset_ = {'targets': dset.targets.data,
             'data': dset.data.loc[dset.targets.data[dset.targets.data.ndet >= 2].index]}
    del dset

    path = "../sim_output/"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created! ", path)

    outputfilename = lensfile[:-4] + '_dset_ia_salt2.pkl'
    pickle.dump(dset_, open(outputfilename, 'wb'))
    print('Saved: ', outputfilename)

    file_index += 1

    ndet_total += sum(det > 2)

    print('ndet_total=', ndet_total)