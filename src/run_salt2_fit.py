def parse_commands():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--dset', default='../sim_output/dset_ia_salt2_10000_0.pkl', type=str, help='dset_file')

    return parser.parse_args()

args = parse_commands()

dset_file = args.dset

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
dset_ = pd.read_pickle(dset_file)
lcpar = pd.read_pickle(dset_file[:-4]+'_lcparams.pkl')

from analysis.lc_salt2_fit import lc_salt2_fit

fitted_params = pd.DataFrame({}, index=dset_['targets'].index)
fitted_params['ndet'] = dset_['targets']['ndet'].values

fitted_params['mb_fit'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['t0lens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['x0lens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['x1lens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['clens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['errt0lens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['errx0lens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['errx1lens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['errclens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['chisqlens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['ndof'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['pherr_mb_fit'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['pherr_t0lens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['pherr_x0lens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['pherr_x1lens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['pherr_clens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['pherr_errt0lens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['pherr_errx0lens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['pherr_errx1lens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['pherr_errclens'] = np.ones(len(dset_['targets'])) * np.nan
fitted_params['pherr_chisqlens'] = np.ones(len(dset_['targets'])) * np.nan

import pickle
import os

if os.path.isfile(dset_file[:-4]+'_fittedparams.pkl')==False:
    for obs_ind in dset_['targets'].index[dset_['targets'].ndet >= 5][:]:
        ind = obs_ind
        lc = dset_['data'].loc[ind]
        lc['snr'] = lc.flux / lc.fluxerr
        low_snr_indeces = lc[lc.snr < 5].index
        lc.drop(index=low_snr_indeces, inplace=True)
        mwebv = dset_['targets'].loc[ind].MWebv  # , dset.targets.data.loc[ind].zlens, dset.targets.data.loc[ind].zlens_phzerr
        zlens = dset_['targets'].loc[ind].z_l
        zlens_phzerr = lcpar.loc[ind].zlens_phzerr

        fitted_params_outfile = dset_file[:-4] + '_salt2fit_ind' + str(obs_ind) + '.pkl'
        if os.path.isfile(fitted_params_outfile):
            out = pd.read_pickle(fitted_params_outfile)
        else:
            out = lc_salt2_fit(lc, mwebv, zlens, doPlot=False)
            pickle.dump(out, open(fitted_params_outfile, 'wb'))

        fitted_params_outfile_phzerr = dset_file[:-4] + '_salt2fitphzerr_ind' + str(obs_ind) + '.pkl'
        if os.path.isfile(fitted_params_outfile_phzerr):
            out_phzerr = pd.read_pickle(fitted_params_outfile_phzerr)
        else:
            out_phzerr = lc_salt2_fit(lc, mwebv, zlens_phzerr, doPlot=False)
            pickle.dump(out_phzerr, open(fitted_params_outfile_phzerr, 'wb'))

        fitted_params['mb_fit'].loc[ind] = out['mb_fit']
        fitted_params['t0lens'].loc[ind] = out['t0lens']
        fitted_params['x0lens'].loc[ind] = out['x0lens']
        fitted_params['x1lens'].loc[ind] = out['x1lens']
        fitted_params['clens'].loc[ind] = out['clens']
        fitted_params['errt0lens'].loc[ind] = out['errt0lens']
        fitted_params['errx0lens'].loc[ind] = out['errx0lens']
        fitted_params['errx1lens'].loc[ind] = out['errx1lens']
        fitted_params['errclens'].loc[ind] = out['errclens']
        fitted_params['chisqlens'].loc[ind] = out['chisqlens']
        fitted_params['ndof'].loc[ind] = out['ndof']
        fitted_params['pherr_mb_fit'].loc[ind] = out_phzerr['mb_fit']
        fitted_params['pherr_t0lens'].loc[ind] = out_phzerr['t0lens']
        fitted_params['pherr_x0lens'].loc[ind] = out_phzerr['x0lens']
        fitted_params['pherr_x1lens'].loc[ind] = out_phzerr['x1lens']
        fitted_params['pherr_clens'].loc[ind] = out_phzerr['clens']
        fitted_params['pherr_errt0lens'].loc[ind] = out_phzerr['errt0lens']
        fitted_params['pherr_errx0lens'].loc[ind] = out_phzerr['errx0lens']
        fitted_params['pherr_errx1lens'].loc[ind] = out_phzerr['errx1lens']
        fitted_params['pherr_errclens'].loc[ind] = out_phzerr['errclens']
        fitted_params['pherr_chisqlens'].loc[ind] = out_phzerr['chisqlens']



    # SAVE

    fitted_params.to_pickle(dset_file[:-4]+'_fittedparams.pkl')
    print('Saved ', dset_file[:-4]+'_fittedparams.pkl')

    # Delete intermediate files
    fitted_params_outfiles_rm = dset_file[:-4] + '_salt2fit*_ind*.pkl'
    os.system('rm '+fitted_params_outfiles_rm )

    print('Deleted: ', fitted_params_outfiles_rm)