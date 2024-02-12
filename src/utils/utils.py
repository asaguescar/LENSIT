# I want to combine the params and the targets
import glob
import pandas as pd
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import interp1d
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')


def combine_target_and_lcparams(globinput = '../sim_output/dset_ia_salt2_10000_?.pkl'):
    dsetfiles = glob.glob(globinput)
    dsets = []
    for dset_ in dsetfiles:
        df_ = pd.read_pickle(dset_)
        lcpar_ = pd.read_pickle(dset_[:-4]+'_lcparams.pkl')
        try:
            fitpar_ = pd.read_pickle(dset_[:-4]+'_fittedparams.pkl')
            dfcomb_ = pd.concat([df_['targets'], lcpar_, fitpar_], axis=1)
        except:
            dfcomb_ = pd.concat([df_['targets'], lcpar_], axis=1)
        dfcomb_ = dfcomb_.loc[:, ~dfcomb_.columns.duplicated()] # to eliminate duplicated columns
        dsets.append(dfcomb_)
    return pd.concat(dsets, ignore_index=True)

def event_rate(z, Rloc=3e4, az=1):
    '''
    Rloc (Gpc-3) is the local event rate and αz parametrizes the red- shift evolution
    '''
    return Rloc * (1+z)**az

def comoving_rate(z, Rloc=3e4, az=1, omega_sky=4*np.pi):
    '''Rsl(< zmax)
    omega_sky : sky area of the survey
    The factor (1 + z)−1 takes account of the time dilation
    R(z) is usually defined as the event rate in the rest frame of the transients'''

    import sys
    sys.path.append('..')
    from src.simulations.simulating_lenses import psl

    def integral_function(z):
        return cosmo.differential_comoving_volume(z).to('Gpc3/sr').value * event_rate(z, Rloc, az) / (1+z) * psl(z)
    return omega_sky * np.array([quad(integral_function, 0, zmax)[0] for zmax in z])

def add_weight_fromuniformz(z, zmax=1.5, Rloc = 2.35e4, alpha= 1., fraction_sky=.75):

    x_cor = np.linspace(0, zmax+0.1, 10000)
    y_cor = np.diff(comoving_rate(x_cor, Rloc=Rloc, az=alpha))
    f_cor = interp1d(np.mean([x_cor[1:], x_cor[:-1]], axis=0), y_cor)

    all_sky_rate = float(comoving_rate([zmax], Rloc=Rloc, az=alpha)) * fraction_sky

    weight =  f_cor(z)
    weight = weight/weight.sum() * all_sky_rate
    return weight
