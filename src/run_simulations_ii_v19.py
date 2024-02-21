# Run The LENSIT pipeline to get the detected lightcurves

def parse_commands():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-z', '--zmax', default=1.5, type=float, help='max redshift')
    parser.add_argument('-s', '--size', default=10, type=int, help='size')
    parser.add_argument('-i', '--index', default=0, type=int, help='index')

    return parser.parse_args()

args = parse_commands()
##############################
# Set up the sims:
z_max = args.zmax
size  = args.size
i     = args.index
##############################

from skysurvey.source import get_sncosmo_sourcenames
from simulations.glsne_target import GLSNe
import sncosmo

KIND = 'SN II'
templates = get_sncosmo_sourcenames(KIND, startswith="v19", endswith="corr")  # all -corr models
for it_ in range(len(templates)):
    print(templates[it_], '>> gl-' + templates[it_])
    source = GLSNe(templates[it_], name='gl-' + templates[it_])
    sncosmo.registry.register(source, force=True)

from simulations.simulating_lenses import sample_lensing_parameters
from astropy.cosmology import Planck18 as cosmo
import pandas as pd

out = sample_lensing_parameters(z_max=z_max, cosmo=cosmo, size=size)
outdf = pd.DataFrame(out)
outlensed = outdf[outdf.Lensed == 1]
del out
del outdf


from simulations.simulating_sne import glsneii_sample

out_ii = glsneii_sample(outlensed.to_dict('list'), cosmo)

import pandas as pd
survey = pd.read_pickle('../input/logs/ztf_survey_p12_allsurvey.pkl')

from skysurvey import ztf
ztf_survey = ztf.ZTF(survey, level='quadrant')
del survey

from simulations.glsne_target import GLSNeII
import numpy as np

ii = pd.DataFrame(out_ii)
ii = ii[ii.Lensed==1]
sample = GLSNeII.from_draw(size=len(ii))
sample.data['template'] = ii['template']
sample.data['z'] = ii['z']
sample.data['ra'] = ii['ra']
sample.data['dec'] = ii['dec']
sample.data['amplitude'] = ii['amplitude']
sample.data.drop(columns=['magabs','magobs', 't0'],inplace=True)

for c in ii.columns:
    sample.data[c] = ii[c]
del ii

# Add t0
starting_date = ztf_survey.data.mjd.min()
ending_date = ztf_survey.data.mjd.max()
sample.data['t0'] = np.random.uniform(starting_date, ending_date, size=len(sample.data['z']))



from skysurvey import DataSet

dset = DataSet.from_targets_and_survey(sample, ztf_survey)
del sample
del ztf_survey

det = dset.get_ndetection()
dset.targets.data['ndet'] = np.nan
dset.targets.data['ndet'].loc[det.index] = det


import pickle

dset_ = {'targets': dset.targets.data,
         'data': dset.data}
del dset

import os
path = "../sim_output/"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created! ", path)

outputfilename = path+'dset_ii_v19_'+str(size)+'_'+str(args.index)+'.pkl'
pickle.dump(dset_, open(outputfilename, 'wb'))
print('Saved: ', outputfilename)
