# Run The LENSIT pipeline to get the detected lightcurves

##############################
# Set up the sims:
z_max = 1.5
size  = 10
i = 0
##############################

from simulations.simulating_lenses import sample_object_parameters
from astropy.cosmology import Planck18 as cosmo
import pandas as pd

out = sample_object_parameters(z_max=z_max, cosmo=cosmo, size=size)
outdf = pd.DataFrame(out)
outlensed = outdf[outdf.Lensed == 1]
del out
del outdf


from simulations.simulating_sne import glsneia_salt2_sample

out_ia_salt2 = glsneia_salt2_sample(outlensed.to_dict('list'), cosmo)


import pandas as pd
survey = pd.read_pickle('../input/logs/ztf_survey_p12_allsurvey.pkl')

from skysurvey import ztf
ztf_survey = ztf.ZTF(survey, level='quadrant')
del survey

from simulations.glsne_target import GLSNeIa_salt2
import numpy as np

# Add t0
starting_date = ztf_survey.data.mjd.min()
ending_date = ztf_survey.data.mjd.max()
out_ia_salt2['t0'] = np.random.uniform(starting_date, ending_date, size=len(out_ia_salt2['z']))

ia_salt2 = pd.DataFrame(out_ia_salt2)
sample = GLSNeIa_salt2.from_data(ia_salt2[ia_salt2.Lensed==1])


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

outputfilename = path+'dset_ia_salt2_'+str(size)+'_'+str(i)+'.pkl'
pickle.dump(dset_, open(outputfilename, 'wb'))
print('Saved: ', outputfilename)
