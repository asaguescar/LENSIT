from utils.utils import combine_target_and_lcparams
import numpy as np

from utils.utils import add_weight_fromuniformz

# and add weights

zmax = 1.5
fraction_sky = .75

# Combine targets lcparams for salt2
df = combine_target_and_lcparams(globinput = '../sim_output/dset_ia_salt2_?0000_?.pkl')
df['ndet_aroundpeak'] = df['g_npoints_aroundpeak']+df['r_npoints_aroundpeak']+df['i_npoints_aroundpeak']
df['dmag'] = 2.5*np.log10(df.mu_total)
Rloc = 2.35e4
alpha= 1.5
df['weight'] = add_weight_fromuniformz(df.z, zmax=zmax, Rloc = Rloc, alpha= alpha, fraction_sky=fraction_sky)
df.to_pickle('../sim_output/dset_ia_salt2_targlcpar.pkl')

# Combine targets lcparams for hsiao
df = combine_target_and_lcparams(globinput = '../sim_output/dset_ia_hsiao_?0000_?.pkl')
df['ndet_aroundpeak'] = df['g_npoints_aroundpeak']+df['r_npoints_aroundpeak']+df['i_npoints_aroundpeak']
df['dmag'] = 2.5*np.log10(df.mu_total)
Rloc = 2.35e4
alpha= 1.5
df['weight'] = add_weight_fromuniformz(df.z, zmax=zmax, Rloc = Rloc, alpha= alpha, fraction_sky=fraction_sky)
df.to_pickle('../sim_output/dset_ia_hsiao_targlcpar.pkl')

# Combine targets lcparams for iip
df = combine_target_and_lcparams(globinput = '../sim_output/dset_iip_?0000_?.pkl')
df['ndet_aroundpeak'] = df['g_npoints_aroundpeak']+df['r_npoints_aroundpeak']+df['i_npoints_aroundpeak']
df['dmag'] = 2.5*np.log10(df.mu_total)
Rloc = 5.52e4
alpha= 2.
df['weight'] = add_weight_fromuniformz(df.z, zmax=zmax, Rloc = Rloc, alpha= alpha, fraction_sky=fraction_sky)
df.to_pickle('../sim_output/dset_iip_targlcpar.pkl')

# Combine targets lcparams for iin
df = combine_target_and_lcparams(globinput = '../sim_output/dset_iin_?0000_?.pkl')
df['ndet_aroundpeak'] = df['g_npoints_aroundpeak']+df['r_npoints_aroundpeak']+df['i_npoints_aroundpeak']
df['dmag'] = 2.5*np.log10(df.mu_total)
Rloc = 5.05e3
alpha= 2.
df['weight'] = add_weight_fromuniformz(df.z, zmax=zmax, Rloc = Rloc, alpha= alpha, fraction_sky=fraction_sky)
df.to_pickle('../sim_output/dset_iin_targlcpar.pkl')

# Combine targets lcparams for ibc
df = combine_target_and_lcparams(globinput = '../sim_output/dset_ibc_?0000_?.pkl')
df['ndet_aroundpeak'] = df['g_npoints_aroundpeak']+df['r_npoints_aroundpeak']+df['i_npoints_aroundpeak']
df['dmag'] = 2.5*np.log10(df.mu_total)
Rloc = 3.33e4
alpha= 2.
df['weight'] = add_weight_fromuniformz(df.z, zmax=zmax, Rloc = Rloc, alpha= alpha, fraction_sky=fraction_sky)
df.to_pickle('../sim_output/dset_ibc_targlcpar.pkl')
