import glob
dsets_list = glob.glob('../sim_output/lenses_*_*_dset_iip_s11-2005lc.pkl')


import sys
sys.path.append('..')
from src.utils import format_skysurvey_outputs

not_observed, not_detected, detectable, identifiable = format_skysurvey_outputs(dsets_list, sntype='iip')


outputs = { 'not_observed': not_observed,
            'not_detected': not_detected,
            'detectable': detectable,
            'identifiable': identifiable}

import pickle
pickle.dump(outputs, open('../sim_output/iip_outputs.pkl', 'wb'))