import pandas as pd

salt2par = pd.read_pickle('../sim_output/dset_ia_salt2_targlcpar.pkl')
salt2par = salt2par[salt2par.ndet>2]
salt2par.to_pickle('../sim_output/dset_ia_salt2_targlcpar_det.pkl')

'''
hsiaopar = pd.read_pickle('../sim_output/dset_ia_hsiao_targlcpar.pkl')
hsiaopar = hsiaopar[hsiaopar.ndet>2]
salt2par.to_pickle('../sim_output/dset_ia_salt2_targlcpar_det.pkl')

iippar = pd.read_pickle('../sim_output/dset_iip_targlcpar.pkl')
iippar = iippar[iippar.ndet>2]
iippar.to_pickle('../sim_output/dset_iip_targlcpar_det.pkl')

iinpar = pd.read_pickle('../sim_output/dset_iin_targlcpar.pkl')
iinpar = iinpar[iinpar.ndet>2]
iinpar.to_pickle('../sim_output/dset_iin_targlcpar_det.pkl')

ibcpar = pd.read_pickle('../sim_output/dset_ibc_targlcpar.pkl')
ibcpar = ibcpar[ibcpar.ndet>2]
ibcpar.to_pickle('../sim_output/dset_ibc_targlcpar_det.pkl')
'''
