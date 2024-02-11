def parse_commands():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--dset', default='../sim_output/dset_ia_salt2_10000_0.pkl', type=str, help='dset_file')
    parser.add_argument('-s', '--sntype', default='salt2', type=str, help='salt2/hsiao/iip/iin/ibc')

    return parser.parse_args()

args = parse_commands()

dset_file = args.dset


import warnings
warnings.filterwarnings('ignore')


if args.sntype == 'salt2':
    from simulations.glsne_target import GLSNeIa_salt2
    mod = GLSNeIa_salt2._TEMPLATE
elif args.sntype == 'hsiao':
    from simulations.glsne_target import GLSNeIa_hsiao
    mod = GLSNeIa_hsiao._TEMPLATE
elif args.sntype == 'iip':
    from simulations.glsne_target import GLSNe_sn2p_2005lc
    mod = GLSNe_sn2p_2005lc._TEMPLATE
elif args.sntype == 'iin':
    from simulations.glsne_target import GLSNe_sn2n
    mod = GLSNe_sn2n._TEMPLATE
elif args.sntype == 'ibc':
    from simulations.glsne_target import GLSNe_sn1bc
    mod = GLSNe_sn1bc._TEMPLATE

import pandas as pd
dset_ = pd.read_pickle(dset_file)

lc_params = pd.DataFrame({}, index=dset_['targets'].index)
lc_params['ndet'] = dset_['targets']['ndet'].values

# We first add the photometric redshift correlated to the predicted photometric error with the true redshift
# This is interesting for inspecting the effect on the infered brightness
from analysis.lcs_parameters import add_photoz_error

lc_params['zlens_phzerr'], lc_params['phzerr'] = add_photoz_error(dset_['targets']['z_l'].values, rel_err=0.15)


# We want to measure the observed time of peak from the lightcurve

# We know from the input simulation. t0! :)
# we want to know the apparent magnitude in g,r and i at t0.

from analysis.lcs_parameters import time_modelpeak
import numpy as np

lc_params['t_peak'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['g_modelpeak'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['r_modelpeak'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['i_modelpeak'] = np.ones(len(dset_['targets'])) * np.nan
for ind in dset_['targets'].index:
    try:
        mod_par = dset_['targets'].loc[ind][['z', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'dt_1', 'dt_2', 'dt_3',
                                             'dt_4', 'x1', 'c', 'x0', 'hostr_v', 'hostebv', 'MWebv', 't0']]
    except:
        mod_par = dset_['targets'].loc[ind][['z', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'dt_1', 'dt_2', 'dt_3',
                                             'dt_4', 'amplitude', 'hostr_v', 'hostebv', 'MWebv', 't0']]
    mod.set(**mod_par)
    lc_params['t_peak'].loc[ind], lc_params['g_modelpeak'].loc[ind], lc_params['r_modelpeak'].loc[ind], lc_params['i_modelpeak'].loc[ind]= time_modelpeak(mod)


# To get colors and peak from the lcs lets share tools with lsst analysis
from analysis.lcs_parameters import color_model

lc_params['t_color_obs']     = np.ones(len(dset_['targets'])) * np.nan
lc_params['g-r_mod']         = np.ones(len(dset_['targets'])) * np.nan
lc_params['g-i_mod']         = np.ones(len(dset_['targets'])) * np.nan
lc_params['r-i_mod']         = np.ones(len(dset_['targets'])) * np.nan
lc_params['t_color_obs']  = lc_params['t_color_obs'].astype(object)
lc_params['g-r_mod']      = lc_params['g-r_mod'].astype(object)
lc_params['g-i_mod']      = lc_params['g-i_mod'].astype(object)
lc_params['r-i_mod']      = lc_params['r-i_mod'].astype(object)
for ind in dset_['targets'].index:
    try:
        mod_par = dset_['targets'].loc[ind][['z', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'dt_1', 'dt_2', 'dt_3',
                                             'dt_4', 'x1', 'c', 'x0', 'hostr_v', 'hostebv', 'MWebv', 't0']]
    except:
        mod_par = dset_['targets'].loc[ind][['z', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'dt_1', 'dt_2', 'dt_3',
                                             'dt_4', 'amplitude', 'hostr_v', 'hostebv', 'MWebv', 't0']]
    mod.set(**mod_par)
    lc_params['t_color_obs'].loc[ind], lc_params['g-r_mod'].loc[ind], lc_params['g-i_mod'].loc[ind], lc_params['r-i_mod'].loc[ind]= color_model(mod, lc_params['t_peak'].loc[ind])

from analysis.lcs_parameters import get_lcs_params

# From here we add rise time, decay time, observerd tpeak and peak mag and we add colors
# Number of points. N_points around peak

lc_params['g_npoints'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['r_npoints'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['i_npoints'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['g_npoints_aroundpeak'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['r_npoints_aroundpeak'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['i_npoints_aroundpeak'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['g_rise_time'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['r_rise_time'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['i_rise_time'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['g_decay_time'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['r_decay_time'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['i_decay_time'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['obs_t_peak'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['g_peak_mag'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['r_peak_mag'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['i_peak_mag'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['g_peak_magerr'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['r_peak_magerr'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['i_peak_magerr'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['g_mag_epochs'] = (np.ones(len(dset_['targets'])) * np.nan).astype(object)
lc_params['r_mag_epochs'] = (np.ones(len(dset_['targets'])) * np.nan).astype(object)
lc_params['i_mag_epochs'] = (np.ones(len(dset_['targets'])) * np.nan).astype(object)
lc_params['g_magerr_epochs'] = (np.ones(len(dset_['targets'])) * np.nan).astype(object)
lc_params['r_magerr_epochs'] = (np.ones(len(dset_['targets'])) * np.nan).astype(object)
lc_params['i_magerr_epochs'] = (np.ones(len(dset_['targets'])) * np.nan).astype(object)
lc_params['firstdet'] = np.ones(len(dset_['targets'])) * np.nan
lc_params['lastdet'] = np.ones(len(dset_['targets'])) * np.nan

for obs_ind in dset_['targets'].index[dset_['targets'].ndet >= 5][:]:
    #print(obs_ind)
    ind = obs_ind

    times_ = dset_['data'].loc[ind].time.values
    mags_ = dset_['data'].loc[ind].zp.values - 2.5 * np.log10(dset_['data'].loc[ind].flux.values)
    magerrs_ = 2.5 / np.log(10) * (dset_['data'].loc[ind].fluxerr.values / dset_['data'].loc[ind].flux.values)
    bands_ = dset_['data'].loc[ind].band.values
    snr_ = dset_['data'].loc[ind].flux.values / dset_['data'].loc[ind].fluxerr.values

    out = get_lcs_params(times_, mags_, magerrs_, bands_, snr_, doPlot=False)
    lc_params['g_npoints'].loc[ind] = out['ztfg_npoints']
    lc_params['r_npoints'].loc[ind] = out['ztfr_npoints']
    lc_params['i_npoints'].loc[ind] = out['ztfi_npoints']
    lc_params['g_npoints_aroundpeak'].loc[ind] = out['ztfg_npoints_aroundpeak']
    lc_params['r_npoints_aroundpeak'].loc[ind] = out['ztfr_npoints_aroundpeak']
    lc_params['i_npoints_aroundpeak'].loc[ind] = out['ztfi_npoints_aroundpeak']
    lc_params['g_rise_time'].loc[ind] = out['ztfg_rise_time']
    lc_params['r_rise_time'].loc[ind] = out['ztfr_rise_time']
    lc_params['i_rise_time'].loc[ind] = out['ztfi_rise_time']
    lc_params['g_decay_time'].loc[ind] = out['ztfg_decay_time']
    lc_params['r_decay_time'].loc[ind] = out['ztfr_decay_time']
    lc_params['i_decay_time'].loc[ind] = out['ztfi_decay_time']
    lc_params['obs_t_peak'].loc[ind] = out['obs_t_peak']
    lc_params['g_peak_mag'].loc[ind] = out['ztfg_peak_mag']
    lc_params['r_peak_mag'].loc[ind] = out['ztfr_peak_mag']
    lc_params['i_peak_mag'].loc[ind] = out['ztfi_peak_mag']
    lc_params['g_peak_magerr'].loc[ind] = out['ztfg_peak_mag_err']
    lc_params['r_peak_magerr'].loc[ind] = out['ztfr_peak_mag_err']
    lc_params['i_peak_magerr'].loc[ind] = out['ztfi_peak_mag_err']
    lc_params['g_mag_epochs'].loc[ind] = out['ztfg_peak_mag_epochs']
    lc_params['r_mag_epochs'].loc[ind] = out['ztfr_peak_mag_epochs']
    lc_params['i_mag_epochs'].loc[ind] = out['ztfi_peak_mag_epochs']
    lc_params['g_magerr_epochs'].loc[ind] = out['ztfg_peak_magerr_epochs']
    lc_params['r_magerr_epochs'].loc[ind] = out['ztfr_peak_magerr_epochs']
    lc_params['i_magerr_epochs'].loc[ind] = out['ztfi_peak_magerr_epochs']
    lc_params['firstdet'].loc[ind] = out['firstdet']
    lc_params['lastdet'].loc[ind] = out['lastdet']




# SAVE

lc_params.to_pickle(dset_file[:-4]+'_lcparams.pkl')
print('Saved ', dset_file[:-4]+'_lcparams.pkl')
