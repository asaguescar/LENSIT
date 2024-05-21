import glob
dsets_list = glob.glob('../sim_output/lenses_*_*_dset_ia_salt2.pkl')

import sys
sys.path.append('..')
from src.utils import format_skysurvey_outputs

not_observed, not_detected, detectable, identifiable = format_skysurvey_outputs(dsets_list, sntype='ia')


ia_outputs = {'not_observed': not_observed,
              'not_detected': not_detected,
              'detectable': detectable,
              'identifiable': identifiable}

import pickle
pickle.dump(ia_outputs, open('../sim_output/ia_outputs.pkl', 'wb'))