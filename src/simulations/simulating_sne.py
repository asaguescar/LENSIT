from skysurvey import target
import numpy as np
import matplotlib.pyplot as plt
import sfdmap
from skysurvey.tools import utils
import sncosmo

def hostdust_Ia(r_v=2., ebv_rate=0.11, size=None):
    hostr_v = r_v * np.ones(size)
    hostebv = np.random.exponential(ebv_rate, size)
    return hostr_v, hostebv


def glsneia_salt2_sample(out, cosmo, doPlot=False):
    z = out['z']
    size = len(z)

    x, pdf = target.snia.SNeIaStretch.nicolas2021()
    x1 = np.random.choice(x, size=size, p=pdf / np.sum(pdf))

    x, pdf = target.snia.SNeIaColor.intrinsic_and_dust()
    c = np.random.choice(x, size=size, p=pdf / np.sum(pdf))

    MB = target.snia.SNeIaMagnitude.tripp1998(x1, c)

    mb_app_unlensed = cosmo.distmod(z).value + MB
    template = sncosmo.Model('salt2')
    m_current = template.source_peakmag("bessellb", "vega")
    x0 = 10. ** (0.4 * (m_current - mb_app_unlensed)) * template.get("x0")
    #x0 = 10. ** (0.4 * (MB - mb_app_unlensed))

    hostr_v, hostebv = hostdust_Ia(size=size)

    # Now we add sky coordinates and dust extinction. We assume uniform in ra-dec
    ra, dec = utils.random_radec(ra_range=[0, 360], dec_range=[-30, 90], size=size)

    mw_sfdmap = sfdmap.SFDMap(mapdir='../input/sfddata-master')
    MWebv = mw_sfdmap.ebv(ra, dec)

    if doPlot:
        plt.figure()
        plt.plot(x, pdf)
        plt.hist(x1, density=True, bins=20)
        plt.xlabel('x1')
        plt.show()

        plt.figure()
        plt.plot(x, pdf)
        plt.hist(c, density=True, bins=20)
        plt.xlabel('c')
        plt.show()

        plt.figure()
        plt.hist(MB, density=True, bins=20)
        plt.xlabel('MB')
        plt.show()

        plt.figure()
        plt.hist(x0, density=True, bins=20)
        plt.xlabel('x0')
        plt.show()

        plt.figure()
        plt.hist(hostebv, density=True, bins=20)
        plt.xlabel('hostebv')
        plt.show()

        plt.figure()
        plt.hist(hostr_v, density=True, bins=20)
        plt.xlabel('hostr_v')
        plt.show()

        plt.figure()
        plt.scatter(ra, dec, c=MWebv)
        plt.colorbar(label='MWebv')
        plt.ylabel('Dec')
        plt.xlabel('Ra')
        plt.show()

    out_ia_salt2 = out.copy()
    out_ia_salt2['x1'] = x1
    out_ia_salt2['c'] = c
    out_ia_salt2['MB'] = MB
    out_ia_salt2['x0'] = x0
    out_ia_salt2['hostr_v'] = hostr_v
    out_ia_salt2['hostebv'] = hostebv
    out_ia_salt2['ra'] = ra
    out_ia_salt2['dec'] = dec
    out_ia_salt2['MWebv'] = MWebv
    out_ia_salt2['MB_lensed'] = out_ia_salt2['MB'] - (2.5 * np.log10(out_ia_salt2['mu_total']))
    out_ia_salt2['MB_lensed_app'] = cosmo.distmod(out_ia_salt2['z']).value + out_ia_salt2['MB_lensed']
    out_ia_salt2['MB_unlensed_app'] = cosmo.distmod(out_ia_salt2['z']).value + out_ia_salt2['MB']

    return out_ia_salt2



def glsneia_hsiao_sample(out, cosmo, doPlot=False, mabs=-19.3, sigmaint=0.10):
    z = out['z']
    size = len(z)

    MB = np.random.normal(loc=mabs, scale=sigmaint, size=len(z))

    mb_app_unlensed = cosmo.distmod(z).value + MB
    template = sncosmo.Model('hsiao')
    m_current = template.source_peakmag("bessellb", "vega")
    amp = 10. ** (0.4 * (m_current - mb_app_unlensed)) * template.get("amplitude")
    #x0 = 10. ** (0.4 * (MB - mb_app_unlensed))

    hostr_v, hostebv = hostdust_Ia(size=size)

    # Now we add sky coordinates and dust extinction. We assume uniform in ra-dec
    ra, dec = utils.random_radec(ra_range=[0, 360], dec_range=[-30, 90], size=size)

    mw_sfdmap = sfdmap.SFDMap(mapdir='../input/sfddata-master')
    MWebv = mw_sfdmap.ebv(ra, dec)

    if doPlot:
        plt.figure()
        plt.hist(MB, density=True, bins=20)
        plt.xlabel('MB')
        plt.show()

        plt.figure()
        plt.hist(amp, density=True, bins=20)
        plt.xlabel('x0')
        plt.show()

        plt.figure()
        plt.hist(hostebv, density=True, bins=20)
        plt.xlabel('hostebv')
        plt.show()

        plt.figure()
        plt.hist(hostr_v, density=True, bins=20)
        plt.xlabel('hostr_v')
        plt.show()

        plt.figure()
        plt.scatter(ra, dec, c=MWebv)
        plt.colorbar(label='MWebv')
        plt.ylabel('Dec')
        plt.xlabel('Ra')
        plt.show()

    out_ = out.copy()
    out_['MB'] = MB
    out_['amplitude'] = amp
    out_['hostr_v'] = hostr_v
    out_['hostebv'] = hostebv
    out_['ra'] = ra
    out_['dec'] = dec
    out_['MWebv'] = MWebv
    out_['MB_lensed'] = out_['MB'] - (2.5 * np.log10(out_['mu_total']))
    out_['MB_lensed_app'] = cosmo.distmod(out_['z']).value + out_['MB_lensed']
    out_['MB_unlensed_app'] = cosmo.distmod(out_['z']).value + out_['MB']

    return out_


def glsneiip_sample(out, cosmo, doPlot=False, mabs=-16.9, sigmaint=1.12):
    z = out['z']
    size = len(z)

    MB = np.random.normal(loc=mabs, scale=sigmaint, size=len(z))

    mb_app_unlensed = cosmo.distmod(z).value + MB
    template = sncosmo.Model('s11-2005lc')
    m_current = template.source_peakmag("bessellb", "vega")
    amp = 10. ** (0.4 * (m_current - mb_app_unlensed)) * template.get("amplitude")

    hostr_v, hostebv = hostdust_Ia(size=size)

    # Now we add sky coordinates and dust extinction. We assume uniform in ra-dec
    ra, dec = utils.random_radec(ra_range=[0, 360], dec_range=[-30, 90], size=size)

    mw_sfdmap = sfdmap.SFDMap(mapdir='../input/sfddata-master')
    MWebv = mw_sfdmap.ebv(ra, dec)

    if doPlot:
        plt.figure()
        plt.hist(MB, density=True, bins=20)
        plt.xlabel('MB')
        plt.show()

        plt.figure()
        plt.hist(amp, density=True, bins=20)
        plt.xlabel('amplitude')
        plt.show()

        plt.figure()
        plt.hist(hostebv, density=True, bins=20)
        plt.xlabel('hostebv')
        plt.show()

        plt.figure()
        plt.hist(hostr_v, density=True, bins=20)
        plt.xlabel('hostr_v')
        plt.show()

        plt.figure()
        plt.scatter(ra, dec, c=MWebv)
        plt.colorbar(label='MWebv')
        plt.ylabel('Dec')
        plt.xlabel('Ra')
        plt.show()

    out_ = out.copy()
    out_['MB'] = MB
    out_['amplitude'] = amp
    out_['hostr_v'] = hostr_v
    out_['hostebv'] = hostebv
    out_['ra'] = ra
    out_['dec'] = dec
    out_['MWebv'] = MWebv
    out_['MB_lensed'] = out_['MB'] - (2.5 * np.log10(out_['mu_total']))
    out_['MB_lensed_app'] = cosmo.distmod(out_['z']).value + out_['MB_lensed']
    out_['MB_unlensed_app'] = cosmo.distmod(out_['z']).value + out_['MB']

    return out_


def glsneiin_sample(out, cosmo, doPlot=False, mabs=-19.05, sigmaint=0.5):
    z = out['z']
    size = len(z)

    MB = np.random.normal(loc=mabs, scale=sigmaint, size=len(z))

    mb_app_unlensed = cosmo.distmod(z).value + MB
    template = sncosmo.Model('nugent-sn2n')
    m_current = template.source_peakmag("bessellb", "vega")
    amp = 10. ** (0.4 * (m_current - mb_app_unlensed)) * template.get("amplitude")

    hostr_v, hostebv = hostdust_Ia(size=size)

    # Now we add sky coordinates and dust extinction. We assume uniform in ra-dec
    ra, dec = utils.random_radec(ra_range=[0, 360], dec_range=[-30, 90], size=size)

    mw_sfdmap = sfdmap.SFDMap(mapdir='../input/sfddata-master')
    MWebv = mw_sfdmap.ebv(ra, dec)

    if doPlot:
        plt.figure()
        plt.hist(MB, density=True, bins=20)
        plt.xlabel('MB')
        plt.show()

        plt.figure()
        plt.hist(amp, density=True, bins=20)
        plt.xlabel('amplitude')
        plt.show()

        plt.figure()
        plt.hist(hostebv, density=True, bins=20)
        plt.xlabel('hostebv')
        plt.show()

        plt.figure()
        plt.hist(hostr_v, density=True, bins=20)
        plt.xlabel('hostr_v')
        plt.show()

        plt.figure()
        plt.scatter(ra, dec, c=MWebv)
        plt.colorbar(label='MWebv')
        plt.ylabel('Dec')
        plt.xlabel('Ra')
        plt.show()

    out_ = out.copy()
    out_['MB'] = MB
    out_['amplitude'] = amp
    out_['hostr_v'] = hostr_v
    out_['hostebv'] = hostebv
    out_['ra'] = ra
    out_['dec'] = dec
    out_['MWebv'] = MWebv
    out_['MB_lensed'] = out_['MB'] - (2.5 * np.log10(out_['mu_total']))
    out_['MB_lensed_app'] = cosmo.distmod(out_['z']).value + out_['MB_lensed']
    out_['MB_unlensed_app'] = cosmo.distmod(out_['z']).value + out_['MB']

    return out_



def glsneibc_sample(out, cosmo, doPlot=False, mabs=-17.51, sigmaint=0.74):
    z = out['z']
    size = len(z)

    MB = np.random.normal(loc=mabs, scale=sigmaint, size=len(z))

    mb_app_unlensed = cosmo.distmod(z).value + MB
    template = sncosmo.Model('nugent-sn1bc')
    m_current = template.source_peakmag("bessellb", "vega")
    amp = 10. ** (0.4 * (m_current - mb_app_unlensed)) * template.get("amplitude")

    hostr_v, hostebv = hostdust_Ia(size=size)

    # Now we add sky coordinates and dust extinction. We assume uniform in ra-dec
    ra, dec = utils.random_radec(ra_range=[0, 360], dec_range=[-30, 90], size=size)

    mw_sfdmap = sfdmap.SFDMap(mapdir='../input/sfddata-master')
    MWebv = mw_sfdmap.ebv(ra, dec)

    if doPlot:
        plt.figure()
        plt.hist(MB, density=True, bins=20)
        plt.xlabel('MB')
        plt.show()

        plt.figure()
        plt.hist(amp, density=True, bins=20)
        plt.xlabel('amplitude')
        plt.show()

        plt.figure()
        plt.hist(hostebv, density=True, bins=20)
        plt.xlabel('hostebv')
        plt.show()

        plt.figure()
        plt.hist(hostr_v, density=True, bins=20)
        plt.xlabel('hostr_v')
        plt.show()

        plt.figure()
        plt.scatter(ra, dec, c=MWebv)
        plt.colorbar(label='MWebv')
        plt.ylabel('Dec')
        plt.xlabel('Ra')
        plt.show()

    out_ = out.copy()
    out_['MB'] = MB
    out_['amplitude'] = amp
    out_['hostr_v'] = hostr_v
    out_['hostebv'] = hostebv
    out_['ra'] = ra
    out_['dec'] = dec
    out_['MWebv'] = MWebv
    out_['MB_lensed'] = out_['MB'] - (2.5 * np.log10(out_['mu_total']))
    out_['MB_lensed_app'] = cosmo.distmod(out_['z']).value + out_['MB_lensed']
    out_['MB_unlensed_app'] = cosmo.distmod(out_['z']).value + out_['MB']

    return out_

from skysurvey.source import get_sncosmo_sourcenames
import random

def glsneii_sample(out, cosmo, doPlot=False, mabs=-16.0, sigmaint=1.3):
    z = out['z']
    size = len(z)

    magabs = np.random.normal(loc=mabs, scale=sigmaint, size=len(z))
    magobs = cosmo.distmod(z).value + magabs

    templates = get_sncosmo_sourcenames('SN II', startswith="v19", endswith="corr")  # all -corr models
    templates = ['gl-' + t for t in templates]
    template = random.choices(templates, k=size)

    amp = np.ones(size)
    for t in np.unique(template):
        ind = np.where(template == t)
        model = sncosmo.Model(t)
        m_current = model.source_peakmag("bessellb", "vega")
        amp[ind] = 10. ** (0.4 * (m_current - magobs[ind])) * model.get("amplitude")

    hostr_v, hostebv = hostdust_Ia(size=size)

    # Now we add sky coordinates and dust extinction. We assume uniform in ra-dec
    ra, dec = utils.random_radec(ra_range=[0, 360], dec_range=[-30, 90], size=size)

    mw_sfdmap = sfdmap.SFDMap(mapdir='../input/sfddata-master')
    MWebv = mw_sfdmap.ebv(ra, dec)

    if doPlot:
        plt.figure()
        plt.hist(MB, density=True, bins=20)
        plt.xlabel('MB')
        plt.show()

        plt.figure()
        plt.hist(amp, density=True, bins=20)
        plt.xlabel('amplitude')
        plt.show()

        plt.figure()
        plt.hist(hostebv, density=True, bins=20)
        plt.xlabel('hostebv')
        plt.show()

        plt.figure()
        plt.hist(hostr_v, density=True, bins=20)
        plt.xlabel('hostr_v')
        plt.show()

        plt.figure()
        plt.scatter(ra, dec, c=MWebv)
        plt.colorbar(label='MWebv')
        plt.ylabel('Dec')
        plt.xlabel('Ra')
        plt.show()

    out_ = out.copy()
    out_['MB'] = magabs
    out_['amplitude'] = amp
    out_['hostr_v'] = hostr_v
    out_['hostebv'] = hostebv
    out_['ra'] = ra
    out_['dec'] = dec
    out_['MWebv'] = MWebv
    out_['MB_lensed'] = out_['MB'] - (2.5 * np.log10(out_['mu_total']))
    out_['MB_lensed_app'] = cosmo.distmod(out_['z']).value + out_['MB_lensed']
    out_['MB_unlensed_app'] = cosmo.distmod(out_['z']).value + out_['MB']
    out_['template'] = template

    return out_