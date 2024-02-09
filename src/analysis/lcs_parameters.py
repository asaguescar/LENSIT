import numpy as np

def add_photoz_error(zl, rel_err=0.15):
    phzerr = zl * rel_err
    zl_phzerr = [np.random.normal(zl[i], phzerr[i]) for i in range(len(zl))]
    return zl_phzerr, phzerr


def modelpeak(mod, t_peak):
    '''
    adding in targets file time of peak and the magnitudes at the true peak extracted from the model
    '''

    g_truepeak = mod.bandmag('ztfg', 'ab', t_peak)
    r_truepeak = mod.bandmag('ztfr', 'ab', t_peak)
    i_truepeak = mod.bandmag('ztfi', 'ab', t_peak)

    return g_truepeak, r_truepeak, i_truepeak
