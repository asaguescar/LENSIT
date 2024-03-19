# Run The LENSIT pipeline to get the detected lightcurves

def parse_commands():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--lensfile', default='../sim_output/lenses_100_0.pkl', type=str, help='lenses file')

    return parser.parse_args()

args = parse_commands()
##############################
# Set up the sims:
lensfile = args.lensfile

##############################

import pandas as pd
outlensed = pd.DataFrame(pd.read_pickle(lensfile))

from astropy.cosmology import Planck18 as cosmo
from simulations.simulating_sne import glsneiin_sample

out_ia_salt2 = glsneiin_sample(outlensed.to_dict('list'), cosmo)
del outlensed

import pandas as pd
survey = pd.read_pickle('../input/logs/ztf_survey_p12_allsurvey.pkl')

from skysurvey import ztf
ztf_survey = ztf.ZTF(survey, level='quadrant')
del survey

from simulations.glsne_target import GLSNe_sn2n
import numpy as np

# Add t0
starting_date = ztf_survey.data.mjd.min()
ending_date = ztf_survey.data.mjd.max()
out_ia_salt2['t0'] = np.random.uniform(starting_date, ending_date, size=len(out_ia_salt2['z']))

ia_salt2 = pd.DataFrame(out_ia_salt2)
del out_ia_salt2
sample = GLSNe_sn2n.from_data(ia_salt2[ia_salt2.Lensed==1])
del ia_salt2

from skysurvey import DataSet

dset = DataSet.from_targets_and_survey(sample, ztf_survey)
del sample
del ztf_survey

det = dset.get_ndetection()
dset.targets.data['ndet'] = np.nan
dset.targets.data['ndet'].loc[det.index] = det


import pickle

dset_ = {'targets': dset.targets.data,
         'data': dset.data.loc[dset.targets.data[dset.targets.data.ndet>=2].index]}
del dset

import os
path = "../sim_output/"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created! ", path)

outputfilename = lensfile[:-4]+'_dset_iin.pkl'
pickle.dump(dset_, open(outputfilename, 'wb'))
print('Saved: ', outputfilename)