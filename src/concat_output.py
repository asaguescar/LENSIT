from utils.utils import combine_target_and_lcparams
import pandas as pd

# Combine targets lcparams for salt2
df = combine_target_and_lcparams(globinput = '../sim_output/dset_ia_salt2_?0000_?.pkl')
df.to_pickle('../sim_output/dset_ia_salt2_targlcpar.pkl')

# Combine targets lcparams for hsiao
df = combine_target_and_lcparams(globinput = '../sim_output/dset_ia_hsiao_?0000_?.pkl')
df.to_pickle('../sim_output/dset_ia_hsiao_targlcpar.pkl')

# Combine targets lcparams for iip
df = combine_target_and_lcparams(globinput = '../sim_output/dset_iip_?0000_?.pkl')
df.to_pickle('../sim_output/dset_iip_targlcpar.pkl')

# Combine targets lcparams for iin
df = combine_target_and_lcparams(globinput = '../sim_output/dset_iin_?0000_?.pkl')
df.to_pickle('../sim_output/dset_iin_targlcpar.pkl')

# Combine targets lcparams for ibc
df = combine_target_and_lcparams(globinput = '../sim_output/dset_ibc_?0000_?.pkl')
df.to_pickle('../sim_output/dset_ibc_targlcpar.pkl')
