import pandas as pd
from skysurvey import ztf
from scipy.constants import c
from astropy import units as u
from astropy.cosmology import Planck18
import numpy as np
from scipy  import interpolate
from scipy.stats import norm

from scipy import stats
from scipy.ndimage import gaussian_filter1d
import sfdmap
import ast


def call_ztf_survey(obslog_path):
    """
    Create a ZTF survey object from observation log data.

    Parameters:
        obslog_path (str): The path to the observation log data file in parquet format.

    Returns:
        survey.ZTF: A ZTF survey object.
    """
    try:
        # Read observation log data
        obslog_df = pd.read_parquet(obslog_path)
        
        # Process the data and create a DataFrame
        ztf_df = pd.DataFrame({
            'expid': obslog_df.expid,
            'mjd': obslog_df.expMJD,
            'band': obslog_df['filter'],
            'skynoise': 1/5 * 10**(-0.4*(obslog_df.maglimcat - obslog_df.zp)),
            'maglim': obslog_df.maglimcat,
            'fieldid': obslog_df.fieldID,
            'rcid': obslog_df.rcid,
            'gain': obslog_df.gain,
            'zp': obslog_df.zp
        })
        
        # Create a ZTF survey object
        ztf_survey = ztf.ZTF(ztf_df, level='quadrant')

        return ztf_survey
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def rate_Ia(z, exp=1.5):
    '''rate per redshift for Ia supernovae: 
    return: rate in Gpc-3 yr-1
    '''
    return 2.35e4 * (1.+z)**exp


def rate_glIa(z, exp=1.5):
    '''rate per redshift for Ia supernovae: 
    return: rate in Gpc-3 yr-1
    '''
    return 2.35e4 * (1.+z)**exp * tau(z,3)

def rate_CC(z, exp=1.5):
    '''rate per redshift for CC supernovae: 
    return: rate in Gpc-3 yr-1
    '''
    return 1.01e5 * (1.+z)**exp


def rate_glCC(z, exp=1.5):
    '''rate per redshift for CC supernovae: 
    return: rate in Gpc-3 yr-1
    '''
    return 1.01e5 * (1.+z)**exp * tau(z,3)

    
def tau(z, mu, F=0.0017):
    '''
    Calculate the optical depth (tau) for a given source redshift, magnification threshold, and dimensionless constant.

    :param z: Source redshift
    :param mu: Magnifications greater than or equal to mu0
    :param F: Dimensionless constant (default is 0.0017)
    :return: Optical depth (tau)
    '''
    c_km_s = (c * u.m).to('km').value
    dc = Planck18.comoving_distance(z).value  # Comoving distance to redshift z (Mpc)
    H0 = Planck18.H0.value  # Hubble constant (km Mpc^-1 s^-1)

    tau_value = F * (dc / (c_km_s * H0 ** (-1))) ** 3 * (2 / mu) ** 2 / 4
    return tau_value


def random_mutotal(z, mu_range=np.linspace(3, 1e2, 1000)):
    '''
    Generate random magnification samples for a given redshift.

    :param zs: Array of source redshifts
    :param mu_range: Array of magnification thresholds (default range from 3 to 100)
    :return: Array of random magnification samples
    '''
    taus = np.array([tau(z_, mu_range) for z_ in z])
    return np.array([float(np.random.choice(mu_range, size=1, p=pdf/np.sum(pdf))) for pdf in taus])


def p_zl(z_lens, z_source):
    """
    Calculate the probability density function (PDF) p(z_l) for lens galaxy redshifts.

    Parameters:
    - z_lens: Lens galaxy redshift.
    - z_source: Source redshift.

    Returns:
    - P: PDF value at z_lens.
    """
    d_ls = Planck18.angular_diameter_distance_z1z2(z_lens, z_source).value
    d_l = Planck18.angular_diameter_distance(z_lens).value
    d_s = Planck18.angular_diameter_distance(z_source).value

    P = (d_ls * d_l) / d_s * (1 + z_lens)**3

    return P

def zlens_from_pdf(z, size=None):
    """
    Calculate lens galaxy redshifts by sampling from a normalized probability density function (PDF) p(z_l).
    
    Parameters:
    - z_source: Source redshift.
    - size: Number of lens galaxy redshift samples to generate (default: None).
    
    Returns:
    - z_lens: Array of sampled lens galaxy redshifts.
    """
    
    # Calculate the PDF values
    z_values = np.linspace(0, z, 100)  # Adjust the number of points as needed
    pdf_values = np.array([p_zl(z_, z) for z_ in z_values])
    
    z_lens = np.concatenate([np.random.choice(z_values.T[i][~np.isnan(pdf_values.T[i])], size=1, p=pdf_values.T[i][~np.isnan(pdf_values.T[i])] / np.nansum(pdf_values.T[i]))  for i in range(len(z_values.T))])
    
    return z_lens


import scipy

def compute_kde(data, weights=None):
    kde = scipy.stats.gaussian_kde(data, weights=weights)
    return kde



def get_imno_td(z, zlens, mu_total,df):
    imno, td_max = [], []

    for i in range(len(z)):
        dzs = .2
        dzl = .2
        dmu = 5

        try:
            z_ = z[i]
            zlens_ = zlens[i]
            mu_total_ = mu_total[i]
            mask_zs = (df.zs < z_ + dzs) & (df.zs > z_ - dzs)
            mask_zl = (df.zl < zlens_ + dzl) & (df.zl > zlens_ - dzl)
            mask_mu = (df.mu_total < mu_total_ + dmu) & (df.mu_total > mu_total_ - dmu)
            mask = mask_zs & mask_zl & mask_mu

            #kde_imno   = scipy.stats.gaussian_kde(            df.imno[mask], weights=df.weights[mask])
            #imno_ = int(kde_imno.resample(1))
            imno_ = float(np.random.choice(df.imno[mask], size=1, p=df.weights[mask]/np.sum(df.weights[mask])))
            if imno_<2: imno_=2
            elif imno_>4: imno_=4
            mask2 = df.imno[mask]==imno_

            kde_td_max = scipy.stats.gaussian_kde(np.log10(df.td_max[mask][mask2]), weights=df.weights[mask][mask2])
            td_max_ = 10**(float(kde_td_max.resample(1)))
        except:
            try:
                z_ = z[i]
                zlens_ = zlens[i]
                mu_total_ = mu_total[i]
                mask_zs = (df.zs > z_ - dzs)
                mask_zl = (df.zl > zlens_ - dzl)
                mask_mu = (df.mu_total > mu_total_ - dmu)
                mask = mask_zs & mask_zl & mask_mu

                #kde_imno   = scipy.stats.gaussian_kde(            df.imno[mask], weights=df.weights[mask])
                #imno_ = int(kde_imno.resample(1))
                imno_ = float(np.random.choice(df.imno[mask], size=1, p=df.weights[mask]/np.sum(df.weights[mask])))
                if imno_<2: imno_=2
                elif imno_>4: imno_=4
                mask2 = df.imno[mask]==imno_

                kde_td_max = scipy.stats.gaussian_kde(np.log10(df.td_max[mask][mask2]), weights=df.weights[mask][mask2])
                td_max_  = 10**(float(kde_td_max.resample(1)))
            except:
                #kde_imno = scipy.stats.gaussian_kde(            df.imno, weights=df.weights)
                #imno_ = int(kde_imno.resample(1))
                imno_ = float(np.random.choice(df.imno, size=1, p=df.weights/np.sum(df.weights)))
                if imno_<2: imno_=2
                elif imno_>4: imno_=4
                mask2 = df.imno==imno_

                kde_td_max = scipy.stats.gaussian_kde(np.log10(df.td_max[mask2]), weights=df.weights[mask2])
                td_max_ = 10**(float(kde_td_max.resample(1)))

        imno.append(imno_)
        td_max.append(td_max_)

    return np.array(imno), np.array(td_max)

def inmo_dt_dist(z, zlens, mu_total,df):
    imno, dt_max = get_imno_td(z, zlens, mu_total, df)
    return imno, dt_max

def redshift_Ia_DG(df, zmax=2, zmin=0, size=None):
    data = df.zs
    weights = df.weights
    kde = compute_kde(data, weights=weights)
    return kde.resample(size)


def dts_dist(imno, dt_max, size=None):
    dt_1 = np.zeros(len(imno))
    dt_2 = dt_max
    dt_3 = np.zeros(len(imno))
    dt_4 = np.zeros(len(imno))
    q3 = np.random.uniform(.1, .9, size=sum(imno>2))
    dt_3[imno>2] = dt_2[imno>2]*q3
    q4 = np.random.uniform(.1, .9, size=sum(imno>3))
    dt_4[imno>3] = dt_4[imno>3]*q4

    return dt_1, dt_2, dt_3, dt_4


def mus_dist(imno, mu_total, size=None):
    q = np.array([np.zeros(4) for i in range(len(imno))])
    for i in np.arange(len(imno))[imno==2]:
        q_ = np.array([np.random.uniform(0, 1) for i in range(2)])
        q_ = q_ / np.sum(q_)
        q[i][0], q[i][1] = q_

    for i in np.arange(len(imno))[imno==3]:
        q_ = np.array([np.random.uniform(0, 1) for i in range(3)])
        q_ = q_ / np.sum(q_)
        q[i][0], q[i][1], q[i][2] = q_

    for i in np.arange(len(imno))[imno==4]:
        q_ = np.array([np.random.uniform(0, 1) for i in range(4)])
        q_ = q_ / np.sum(q_)
        q[i][0], q[i][1], q[i][2], q[i][3] = q_

    mu_1 = mu_total * q.T[0]
    mu_2 = mu_total * q.T[1]
    mu_3 = mu_total * q.T[2]
    mu_4 = mu_total * q.T[3]

    return mu_1, mu_2, mu_3, mu_4

def x1_pdf(mu1=0.33, sigma1=0.64, mu2=-1.50, sigma2=0.58, a=0.45, xx=np.arange(-4,4,.005), K=0.87, size=None): 
    '''
    Pdf Strech per redshift pdf from paper Nicolas 2021
    equation 2. 
    '''
    mode1 = norm.pdf(xx, loc=mu1, scale=sigma1)
    mode2 = norm.pdf(xx, loc=mu2, scale=sigma2)

    #fysne = 1/( 1/K * (1+z)**(-2.8)  + 1 )
    fysne = 0.5
    pdf = fysne*mode1 + (1-fysne)*(a*mode1 + (1-a)*mode2)

    return xx, pdf

def x1_dist(z, mu1=0.33, sigma1=0.64, mu2=-1.50, sigma2=0.58, a=0.45, xx=np.arange(-4,4,.005), K=0.87, size=None): 
    '''
    Strech per redshift pdf from paper Nicolas 2021
    equation 2. 
    '''
    mode1 = norm.pdf(xx, loc=mu1, scale=sigma1)
    mode2 = norm.pdf(xx, loc=mu2, scale=sigma2)

    x1s = []
    for i in range(len(z)):
        fysne = 1/( 1/K * (1+z[i])**(-2.8)  + 1 )
        pdf = fysne*mode1 + (1-fysne)*(a*mode1 + (1-a)*mode2)
        x1s.append( np.random.choice(xx, size=size, p=pdf/pdf.sum()) )

    return x1s



def c_pdf(xx=np.arange(-.3,1,.001), cint=-0.05, sigmaint=0.05, tau=0.1):
    '''
    Pdf sn intrinsicc and dust color
    '''
    # exponential decay center on cint
    expon = stats.expon.pdf(xx, loc=-0.05, scale=0.1)
    # applying gaussian filtering
    #  - which require sigmaint in pixel.
    sigmaint_inpix = sigmaint/(xx[1]-xx[0]) # assuming constant step
    pdf = gaussian_filter1d(expon, sigmaint_inpix)
    
    return xx, pdf

def magabs_Ia(x1, c, mabs=-19.3, sigmaint=0.10,
                        alpha=-0.14, beta=3.15, gamma=0.1):
    """
    tripp1998
    """
    mabs = np.random.normal(loc=mabs, scale=sigmaint, size=len(x1))
    mabs_notstandard = mabs + (x1*alpha + c*beta)
    return mabs_notstandard

def magabs_Iahsiao(z, mabs=-19.3, sigmaint=0.10):
    """
    normal distribution 
    """
    mabs = np.random.normal(loc=mabs, scale=sigmaint, size=len(z))
    return mabs

def magabs_IIP(z, mabs=-16.9, sigmaint=1.12):
    """
    normal distribution 
    """
    mabs = np.random.normal(loc=mabs, scale=sigmaint, size=len(z))
    return mabs


def magabs_IIn(z, mabs=-19.05, sigmaint=0.5):
    """
    normal distribution 
    """
    mabs = np.random.normal(loc=mabs, scale=sigmaint, size=len(z))
    return mabs

def magabs_Ibc(z, mabs=-17.51, sigmaint=0.74):
    """
    normal distribution 
    """
    mabs = np.random.normal(loc=mabs, scale=sigmaint, size=len(z))
    return mabs


def magabs_magnified(magabs, mu_total, size=None):
    dm = 2.5 * np.log10(mu_total)
    return magabs - dm


def fromsfdmaps(ra, dec):
    '''
    https://github.com/kbarbary/sfddata
    '''
    m = sfdmap.SFDMap(mapdir='/Users/anasaguescarracedo/Dropbox/PhD/glsne/sfddata')
    return m.ebv(ra, dec)

def hostdust_Ia(r_v=2., ebv_rate=0.11, size=None):
    hostr_v = r_v * np.ones(size)
    hostebv = np.random.exponential(ebv_rate, size)
    return hostr_v, hostebv


def add_to_dset_run(dset_targets, dset_data, lensed=True):
    if lensed==True:   
        dset_targets.data['dt_max'] = np.max([dset_targets.data['dt_1'],dset_targets.data['dt_2'], dset_targets.data['dt_3'],dset_targets.data['dt_4']], axis=0) -  np.min([dset_targets.data['dt_1'],dset_targets.data['dt_2'],           dset_targets.data['dt_3'],dset_targets.data['dt_4']], axis=0)

        dset_targets.data['magabs_zlens'] = dset_targets.data['magobs'] - Planck18.distmod(dset_targets.data['zlens'].values).value

    dset_data['snr'] = dset_data['flux'] / dset_data['fluxerr']
    dset_data['mag'] = dset_data['zp'] - 2.5 * np.log10(dset_data['flux'])
    dset_data['magerr'] = 2.5 / np.log(10) * (dset_data['fluxerr'] / dset_data['flux'])
    dset_targets.data['ndet'] = np.ones(len(dset_targets.data)) * np.nan
    dset_targets.data['firstdet'] = np.ones(len(dset_targets.data)) * np.nan
    dset_targets.data['lastdet'] = np.ones(len(dset_targets.data)) * np.nan
    dset_targets.data['dt_5s'] = np.ones(len(dset_targets.data)) * np.nan
    obsind = dset_targets[dset_targets.ndet>=0].index
    print(obsind)
    for j in obsind:
        dset_targets.data['ndet'].loc[j] = sum(dset_data['snr'].loc[j] > 5)
        if dset_targets.data['ndet'].loc[j] > 0:
            dset_targets.data['firstdet'].loc[j]= sorted(dset_data.time.loc[j][dset_data.snr.loc[j] > 5])[0]
            dset_targets.data['lastdet'].loc[j] = sorted(dset_data.time.loc[j][dset_data.snr.loc[j] > 5])[-1]
            dset_targets.data['dt_5s'].loc[j]   = dset_targets.data['lastdet'].loc[j] - dset_targets.data['firstdet'].loc[j]

    dset_targets.data['index'] = dset_targets.data.index
    #dset.targets.data['index_dset'] = (np.ones(len(dset.targets.data)) * i).astype(int)

    return dset_targets, dset_data


def add_photoz_error(zl, rel_err=0.15):
    phzerr = zl * rel_err
    zl_phzerr = [np.random.normal(zl[i], phzerr[i]) for i in range(len(zl))]
    return zl_phzerr

def time_modelpeak(mod):
    ''' 
    adding in targets file time of peak and the magnitudes at the true peak extracted from the model
    '''
    wave = np.linspace(mod.minwave(), mod.maxwave(), 100)
    i, j = mod.mintime(), mod.maxtime()
    n = int(j - i)
    time = np.linspace(i, j, n)
    
    t_peak = time[np.argmax(np.max(mod.flux(time, wave), axis=1))]
    g_truepeak  = mod.bandmag('ztfg', 'ab', t_peak)
    r_truepeak  = mod.bandmag('ztfr', 'ab', t_peak)
    i_truepeak  = mod.bandmag('ztfi', 'ab', t_peak)

    return t_peak, g_truepeak, r_truepeak, i_truepeak

def color_model(mod, t_peak):
    '''
    We extract the color lightcurve from the observer frame epochs -20:50:5 using the Transient template. 
    '''
    t_array = np.arange(-20, 50.1, 5)
    g  = mod.bandmag('ztfg', 'ab', t_array+t_peak)
    r  = mod.bandmag('ztfr', 'ab', t_array+t_peak)
    i  = mod.bandmag('ztfi', 'ab', t_array+t_peak)
    return t_array, g-r, g-i, r-i


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d



def quadratic(x, d, a, b, c):
    return d * x**3 + a * x**2 + b * x + c 


def get_lcs_params(times_, mags_, magerrs_, bands_, snr_, doPlot=False, survey_bands = ['ztfg', 'ztfr', 'ztfi'], snr_threshold=5., degree = 2, brightest_point=False, polynomial_peak=True):
    '''
    In this function we provide the time, magnitude, magnitude error, bandpassed and the signal to noise ratio of the lightcurve data points. 
    As an output it gives the infered observer peak magnitude and the colors. 
    '''
    out_ = {}
    
    # We first clean the low snr datapoints: 
    times, mags, magerrs, bands, snr = times_[snr_>snr_threshold], mags_[snr_>snr_threshold], magerrs_[snr_>snr_threshold], bands_[snr_>snr_threshold], snr_[snr_>snr_threshold]
    ind_sort = np.argsort(times)
    times, mags, magerrs, bands, snr = times[ind_sort], mags[ind_sort], magerrs[ind_sort], bands[ind_sort], snr[ind_sort]

    # We take the peak from the band with more data points. 
    ii = np.argmax([sum(bands==bb) for bb in survey_bands])
    band_mask = bands == survey_bands[ii]
    #print(survey_bands[ii])

    try:
        # fit quadratic function
        params, covariance = curve_fit(quadratic, times[band_mask], mags[band_mask], sigma = magerrs[band_mask])
        # Create a polynomial function using the coefficients
        polynomial = np.poly1d(params)
        
        x_range = np.linspace(min(times[band_mask]), max(times[band_mask]), 100)
        y_fit = polynomial(x_range)
        
        m_peak_ = y_fit[np.argmin(y_fit)]
        t_peak = x_range[np.argmin(y_fit)]

        weighted_residuals = abs((mags[band_mask] - polynomial(times[band_mask])) /  magerrs[band_mask])
        mean_weighted_residuals = np.mean(weighted_residuals)

        doSpline = False
        if abs(mean_weighted_residuals)>5:
            m_peak_ = np.nanmin(mags[band_mask])   # brightest point
            t_peak = float(times[band_mask][np.where(mags[band_mask]==m_peak_)[0]])
            doSpline = True
    except:
        m_peak_ = np.nanmin(mags[band_mask])   # brightest point
        t_peak = float(times[band_mask][np.where(mags[band_mask]==m_peak_)[0]])

    if doPlot:
        plt.figure()
        plt.errorbar(times[band_mask], mags[band_mask], yerr=magerrs[band_mask])
        #plt.plot(x_range, y_fit)
        plt.show()

    out_['obs_t_peak'] = t_peak
    out_['firstdet'] = min(times)
    out_['lastdet']  = max(times)
    # Get magnitudes from each band using polynomial with the same time, so we can extract the colors. 
    for bb in survey_bands:
        t = times[bands==bb] 
        m = mags[bands==bb]
        merr = magerrs[bands==bb]
        sn = snr[bands==bb]
        
        if ('array' in str(type(m)))==True and sum(~np.isnan(m))>0:
            m_i = m[~np.isnan(m)][0]   # magnitude from the first detection
            m_f = m[~np.isnan(m)][-1]  # magnitude of the last detection

            t_i = t[~np.isnan(m)][0]   # Time of first detection
            t_f = t[~np.isnan(m)][-1]  # Time of last detection
            
            x_range = np.linspace(t_i, t_f, 100)
            #print(m_i, m_f, t_i, t_f)
            if doPlot:
                plt.figure()
                plt.errorbar(t, m, yerr=merr, fmt='o', color='b', label='Experimental Data')

            if len(t)>3: # If more than 3 points we do polynomial fit
                params, covariance = curve_fit(quadratic, t, m, sigma=merr)
                polynomial = np.poly1d(params)
                y_fit = polynomial(x_range)
                if doSpline:
                    ff = interp1d(t, m,  bounds_error=False) # magnitude function at any epoch
                else:
                    ff = interp1d(x_range, y_fit,  bounds_error=False) # magnitude function at any epoch

                
                params, covariance = curve_fit(quadratic, t, m+merr, sigma=merr)
                polynomial = np.poly1d(params)
                if doSpline:
                    ff_up = interp1d(t, m+merr,  bounds_error=False)
                    y_fit_upp = ff_up(x_range)
                else:
                    y_fit_upp = polynomial(x_range)
                params, covariance = curve_fit(quadratic, t, m-merr, sigma=merr)
                polynomial = np.poly1d(params)
                if doSpline:
                    ff_low = interp1d(t, m-merr,  bounds_error=False)
                    y_fit_low = ff_low(x_range)
                else:
                    y_fit_low = polynomial(x_range)
                y_fit = np.array([ np.max([y_fit_upp[i],y_fit_low[i]])-np.min([y_fit_upp[i],y_fit_low[i]])  for i in range(len(x_range))])
                ffmerr = interp1d(x_range, y_fit/2,  bounds_error=False) # Funcction for magnitude errors
                if doPlot:
                    plt.plot(x_range, y_fit_upp, '--', label='Upp bound')
                    plt.plot(x_range, y_fit_low, ':', label='Low bound')
                    plt.plot(x_range, ff(x_range), 'r-', label='Fit')
            elif  len(t)>1:
                ff     = interp1d(t, m   ,  bounds_error=False)
                ffmerr = interp1d(t, merr,  bounds_error=False)
                if doPlot:
                    plt.plot(x_range, ff(x_range), 'r-', label='Fit')
                    plt.plot(x_range, ff(x_range)+ffmerr(x_range), '--', label='Upp bound')
                    plt.plot(x_range, ff(x_range)-ffmerr(x_range), ':', label='Low bound')
            else: 
                ff     = interp1d([-20, 50], [np.nan, np.nan],  bounds_error=False)
                ffmerr = interp1d([-20, 50], [np.nan, np.nan],  bounds_error=False)
                
            m_peak = ff(t_peak)
            
            if doPlot:
                plt.errorbar(t_peak, m_peak, yerr=ffmerr(t_peak), fmt='o', color='r', label='peak')

                plt.xlabel('t')
                plt.ylabel('m '+bb)
                plt.legend()
                plt.show()


            out_[bb+'_npoints'] = np.array(sum(~np.isnan(m)))
            epoch = t - t_peak
            mask_peak = (epoch < 20)&(epoch > -10)
            out_[bb+'_npoints_aroundpeak'] = np.array(sum(~np.isnan(m[mask_peak])))
            out_[bb+'_rise_time'] = t_peak - t_i
            if out_[bb+'_rise_time']>1:
                out_[bb+'_rise_rate'] = (m_peak - m_i) / out_[bb+'_rise_time']
                if out_[bb+'_rise_rate']>0:
                    out_[bb+'_rise_rate'] = np.nan
            else:
                out_[bb+'_rise_rate'] = np.nan
                out_[bb+'_rise_time'] = np.nan
            out_[bb+'_decay_time'] = t_f - t_peak
            if out_[bb+'_decay_time']>1:
                out_[bb+'_decay_rate'] = (m_f - m_peak) / out_[bb+'_decay_time']
                if out_[bb+'_decay_rate']<0:
                    out_[bb+'_decay_rate'] = np.nan
            else:
                out_[bb+'_decay_rate'] = np.nan
                out_[bb+'_decay_time'] = np.nan
            out_[bb+'_peak_mag'] = m_peak
            out_[bb+'_peak_mag_minus7'] = ff(t_peak-7)
            out_[bb+'_peak_mag_plus7'] = ff(t_peak+7)
            out_[bb+'_peak_mag_minus14'] = ff(t_peak-14)
            out_[bb+'_peak_mag_plus14'] = ff(t_peak+14)
            out_[bb+'_peak_mag_err'] = ffmerr(t_peak)
            out_[bb+'_peak_mag_minus7_err'] = ffmerr(t_peak-7)
            out_[bb+'_peak_mag_plus7_err'] = ffmerr(t_peak+7)
            out_[bb+'_peak_mag_minus14_err'] = ffmerr(t_peak-14)
            out_[bb+'_peak_mag_plus14_err'] = ffmerr(t_peak+14)

            t_color = np.arange(-20, 50.1, 5) + t_peak
            out_[bb+'_peak_mag_epochs'] = ff(t_color)
            out_[bb+'_peak_magerr_epochs'] = ffmerr(t_color)
        else:
            out_[bb+'_npoints'] =                  sum(~np.isnan(m))
            epoch = t - t_peak
            mask_peak = (epoch < 20)&(epoch > -10)
            out_[bb+'_npoints_aroundpeak'] = np.array(sum(~np.isnan(m[mask_peak])))
            out_[bb+'_rise_time'] =                np.nan
            out_[bb+'_rise_rate'] =                np.nan
            out_[bb+'_decay_time'] =               np.nan
            out_[bb+'_decay_rate'] =               np.nan
            out_[bb+'_peak_mag'] =                 np.nan
            out_[bb+'_peak_mag_minus7'] =          np.nan
            out_[bb+'_peak_mag_plus7'] =           np.nan
            out_[bb+'_peak_mag_minus14'] =         np.nan
            out_[bb+'_peak_mag_plus14'] =          np.nan
            out_[bb+'_peak_mag_err'] =             np.nan
            out_[bb+'_peak_mag_minus7_err'] =      np.nan
            out_[bb+'_peak_mag_plus7_err'] =       np.nan
            out_[bb+'_peak_mag_minus14_err'] =     np.nan
            out_[bb+'_peak_mag_plus14_err'] =      np.nan
            out_[bb+'_peak_mag_epochs'] = np.nan
            out_[bb+'_peak_magerr_epochs'] = np.nan
    # colors 
    #out_['colors'] = {}
    bands_ =  survey_bands
    for cc1 in range(len(bands_)):
        c1 = bands_[cc1]
        for cc2 in range(cc1+1, len(bands_)):
            c2 = bands_[cc2]
            out_[c1+'-'+c2] = np.array(out_[c1+'_peak_mag']) - np.array(out_[c2+'_peak_mag'])
            out_[c1+'-'+c2+'_err'] = np.sqrt(np.array(out_[c1+'_peak_mag_err'])**2 + np.array(out_[c2+'_peak_mag_err'])**2)

            out_[c1+'-'+c2+'_minus7'] = np.array(out_[c1+'_peak_mag_minus7']) - np.array(out_[c2+'_peak_mag_minus7'])
            out_[c1+'-'+c2+'_minus7_err'] = np.sqrt(np.array(out_[c1+'_peak_mag_minus7_err'])**2 + np.array(out_[c2+'_peak_mag_minus7_err'])**2)

            out_[c1+'-'+c2+'_plus7'] = np.array(out_[c1+'_peak_mag_plus7']) - np.array(out_[c2+'_peak_mag_plus7'])
            out_[c1+'-'+c2+'_plus7_err'] = np.sqrt(np.array(out_[c1+'_peak_mag_plus7_err'])**2 + np.array(out_[c2+'_peak_mag_plus7_err'])**2)
            
            out_[c1+'-'+c2+'_minus14'] = np.array(out_[c1+'_peak_mag_minus14']) - np.array(out_[c2+'_peak_mag_minus14'])
            out_[c1+'-'+c2+'_minus14_err'] = np.sqrt(np.array(out_[c1+'_peak_mag_minus14_err'])**2 + np.array(out_[c2+'_peak_mag_minus14_err'])**2)

            out_[c1+'-'+c2+'_plus14'] = np.array(out_[c1+'_peak_mag_plus14']) - np.array(out_[c2+'_peak_mag_plus14'])
            out_[c1+'-'+c2+'_plus14_err'] = np.sqrt(np.array(out_[c1+'_peak_mag_plus14_err'])**2 + np.array(out_[c2+'_peak_mag_plus14_err'])**2)

    if doPlot:
        plt.figure()
        #plt.errorbar(times, mags, yerr=magerrs, linestyle='none', marker='o')
        for b in bands_:
            plt.errorbar(times[bands==b], mags[bands==b], yerr=magerrs[bands==b], linestyle='none', marker='o', label=b)
        #plt.scatter(times, mags, c=bands)
        plt.plot([t_peak], [m_peak_], marker='x', markersize=50, color='black')
        plt.gca().invert_yaxis()
        #plt.savefig()
        plt.legend(loc=0)
        plt.show()
        
    return out_

import sncosmo
def get_saltmodel(mwebv=None):
    """ SALT2 model incl dust correction """
    dust = sncosmo.F99Dust()
    model = sncosmo.Model("salt2", effects=[dust],
                          effect_names=['mw'],
                          effect_frames=['obs'])
    if mwebv is not None:
        model.set(mwebv=mwebv)

    return model


def fit_lc(data, mwebv=0.1558, redshift=0.35440, modelcov=False, **kwargs):
    """ fit a lightcurve given a pandas Dataframe """

    # get the salt2 + MW dust model.
    model = get_saltmodel(mwebv=mwebv)
    model.set(z=redshift)

    # data as sncosmo input.
    parameters = ['t0', 'x0', 'x1', 'c']

    # fit salt
    #print(model)
    result, fitted_model = sncosmo.fit_lc(data, model, parameters, modelcov=modelcov, **kwargs)

    return result, fitted_model
    
def lc_salt2_fit(lc, mwebv, zlens, doPlot=True, filename=''):
    from astropy.table import Table
    ''' 
    Fit salt2 model to a lightcurve
    '''
    out = {}
    data = Table.from_pandas(lc[['time', 'zp', 'zpsys',
                                         'band', 'flux', 'fluxerr']])
    try:
        result, fitted_model = fit_lc(data, mwebv=mwebv, redshift=zlens)
        if doPlot:
            import matplotlib.pyplot as plt
            _ = sncosmo.plot_lc(data, model=fitted_model, errors=result.errors)
            plt.savefig('plots/'+filename)
            plt.close()
        
        if result.ndof != 0:
            out['mb_fit'] = float(fitted_model.source_peakabsmag('bessellb', 'ab'))
            out['t0lens'] = float(fitted_model.parameters[1])
            out['x0lens'] = float(fitted_model.parameters[2])
            out['x1lens'] = float(fitted_model.parameters[3])
            out['clens'] = float(fitted_model.parameters[4])
            out['errt0lens'] = result.errors['t0']
            out['errx0lens'] = result.errors['x0']
            out['errx1lens'] = result.errors['x1']
            out['errclens'] = result.errors['c']
            out['chisqlens'] = result.chisq / result.ndof
            out['ndof'] = result.ndof
        elif result.ndof == 0:
            out['mb_fit'] = float(fitted_model.source_peakabsmag('bessellb', 'ab'))
            out['t0lens'] = float(fitted_model.parameters[1])
            out['x0lens'] = float(fitted_model.parameters[2])
            out['x1lens'] = float(fitted_model.parameters[3])
            out['clens'] = float(fitted_model.parameters[4])
            out['errt0lens'] = result.errors['t0']
            out['errx0lens'] = result.errors['x0']
            out['errx1lens'] = result.errors['x1']
            out['errclens'] = result.errors['c']
            out['chisqlens'] = 999
            out['ndof'] = 999
    except:
        out['mb_fit'] = 999
        out['t0lens'] = 999
        out['x0lens'] = 999
        out['x1lens'] = 999
        out['clens']  = 999
        out['errt0lens'] = 999
        out['errx0lens'] = 999
        out['errx1lens'] = 999
        out['errclens']  = 999
        out['chisqlens'] = 999
        out['ndof'] = 999

    import pickle
    pickle.dump(out, open('dset_output/salt2fit/lc_salt2_fit_'+str(filename)+'.pkl', 'wb'))
    return out


def detection_criteria(lc):
    '''
    For detection ccriteria we require more than 5 data point with SNR more than 5 around peak: meaning between -10 and +20 days in any band.   
    We also want this detections to happened at least during 5 days. 
    We also want them to look brighter than a 1a SN. mb<-19.5 from the salt2 fit
    '''
    ndetaroundpeak_mask = (lc.g_npoints_aroundpeak + lc.r_npoints_aroundpeak + lc.i_npoints_aroundpeak) > 5
    dt_5s_mask = (lc.firstdet - lc.lastdet) > 5
    mb_fit_mask = lc.mb_fit < -19.5 # We assume here the mb_fit with the true zlens. Then we explore the uncertaintiesand potential misses from pherr

    return ndetaroundpeak_mask & dt_5s_mask & mb_fit_mask  


def sample_with_kde(data, weights, size=1000):
    """
    Sample data from a KDE distribution with optional weights.

    Args:
        data (array-like): Input data.
        weights (array-like): Weights associated with the data.
        size (int): Number of samples to generate.

    Returns:
        numpy.ndarray: Sampled data.
    """
    if len(data) == 0:
        return np.array([])  # Return an empty array if the data is empty
    kde = stats.gaussian_kde(data, weights=weights, bw_method=0.0)
    return kde.resample(size)[0]

def zs_from_DG(df, zmax=2, zmin=0, size=1000):
    # Create a KDE for zs
    return sample_with_kde(df.zs, df.weights, size)

def sample_from_DG(z, df, size=1000, step=0.2):
    """
    Sample and organize data based on a simulation process.

    Args:
        df (pd.DataFrame): Input DataFrame containing required columns.
        size (int): Sample size.
        step (float): Step size for binning.

    Returns:
        dict: A dictionary containing the sampled data for 'zs', 'zl', 'mu_total', 'imno', and 'td_max'.
    """

    dg_par = {
        'zs': [],
        'zl': [],
        'mu_total': [],
        'imno': [],
        'td_max': [],
        "mu_1": [],  "mu_2": [],  "mu_3": [],  "mu_4": [], 
        "dt_1": [],  "dt_2": [],  "dt_3": [],  "dt_4": [],
        "mb" : [], "angsep_max":[], "amplitude":[]
        }

    # Create a KDE for zs
    dg_par['zs'] = z
    
    # Define parameters for the loop
    dzs = step / 2

    n, b = np.histogram(dg_par['zs'], bins=np.arange(dg_par['zs'].min(), dg_par['zs'].max() + step, step))
    zs_m_range =  ((b[1:] + b[:-1])*.5) [n>0]
    zs_n_ = n[n>0]
    for i in range(len(zs_m_range)):
        zs_m = zs_m_range[i]
        mask_zs = (df.zs > (zs_m - dzs)) & (df.zs <= (zs_m + dzs))
        size1 =  zs_n_[i]  # The size would be the number of elements in this cut from the previous simulated ZS
        
        # Simulate zl and extend the dictionary
        zl_sim = sample_with_kde(df.zl[mask_zs], df.weights[mask_zs], size1)
        dg_par['zl'].extend(zl_sim)

        n, b = np.histogram(zl_sim, bins=np.arange(0, np.max(zl_sim) + step, step))
        zl_m_range =  ((b[1:] + b[:-1])*.5) [n>0]
        zl_n_ = n[n>0]
        for i in range(len(zl_m_range)):
            zl_m = zl_m_range[i]
            mask2 = (df.zl>(zl_m - dzs)) & (df.zl<=(zl_m + dzs))
            size2 =  zl_n_[i]
            if size2==0:
                print(size2)
            
            # Simulate mu_total and extend the dictionary
            mutot_sim = sample_with_kde(df.mu_total[mask_zs & mask2], df.weights[mask_zs & mask2], size2)
            dg_par['mu_total'].extend(mutot_sim)
            
            
            # Check if mutot_sim is empty before computing the minimum and maximum
            if not np.any(mutot_sim):
                print('mutot_sim is empty!')
                #continue  # Skip further processing
            
            n, b = np.histogram(mutot_sim, bins=np.logspace(0, np.log10(mutot_sim.max()+.1), 20))
            mutot_m_ =  ((b[1:] + b[:-1])*.5) [n>0]
            mutot_n_ = n[n>0]
            for i in range(len(mutot_m_)):
                mutot_m = mutot_m_[i]
            
                    
                if i == 0: dmu_min = (mutot_m - 1)*.5
                else:  dmu_min = (mutot_m - mutot_m_[i-1])*.5
                if i == len(mutot_m_)-1: dmu_max = (np.max(b) - mutot_m_[i])*.5
                else:  dmu_max = (mutot_m_[i+1] - mutot_m)*.5
                mask3 = (df.mu_total>(mutot_m - dmu_min)) & (df.mu_total<=(mutot_m + dmu_max))
                size3 =  mutot_n_[i]

                imno_sim_ = np.random.choice(df.imno[mask_zs+mask2+mask3], size=size3, p=df.weights[mask_zs+mask2+mask3]/np.sum(df.weights[mask_zs+mask2+mask3]))
                imno_sim_[imno_sim_<2] = 2
                imno_sim_[imno_sim_>4] = 4
                dg_par['imno'].extend( imno_sim_ )
                
                for imno_m in set(imno_sim_):
                    mask4 = df.imno == imno_m
                    size4 = sum( imno_sim_==imno_m )
                    tdmax_sim_ = 10**( stats.gaussian_kde(np.log10(df.td_max[mask_zs+mask2+mask3+mask4]), weights=df.weights[mask_zs+mask2+mask3+mask4], bw_method=0.0).resample( size4 )[0] )
                    dg_par['td_max'].extend( tdmax_sim_ )
    
                    # Get individual mus: 
                    dist_indeces = np.random.choice(df.index[mask_zs+mask2+mask3+mask4][df.imno[mask_zs+mask2+mask3+mask4] == imno_m], size=size4, p=df.weights[mask_zs+mask2+mask3+mask4][df.imno[mask_zs+mask2+mask3+mask4] == imno_m]/np.sum(df.weights[mask_zs+mask2+mask3+mask4][df.imno[mask_zs+mask2+mask3+mask4] == imno_m]))
                    mus = np.array([ast.literal_eval(i) for i in df.mu_ind.loc[dist_indeces].values])
                    dts = np.array([ast.literal_eval(i) for i in df.td_ind.loc[dist_indeces].values])
                    add_cols = 4 - mus.shape[1]
                    mus_1, mus_2, mus_3, mus_4 = np.vstack((mus.T, np.zeros((add_cols, mus.T.shape[1]))))
                    dts_1, dts_2, dts_3, dts_4 = np.vstack((dts.T, np.zeros((add_cols, dts.T.shape[1]))))
                    
                    dg_par['mu_1'].extend( mus_1 )
                    dg_par['mu_2'].extend( mus_2 )
                    dg_par['mu_3'].extend( mus_3 )
                    dg_par['mu_4'].extend( mus_4 )
                    dg_par['dt_1'].extend( dts_1 )
                    dg_par['dt_2'].extend( dts_2 )
                    dg_par['dt_3'].extend( dts_3 )
                    dg_par['dt_4'].extend( dts_4 )

                    dg_par["mb"].extend( df.mb.loc[dist_indeces].values )
                    dg_par["angsep_max"].extend( df.angsep_max.loc[dist_indeces].values )
                    dg_par["amplitude"].extend( df.amplitude.loc[dist_indeces].values )

    return dg_par['zl'], dg_par['mu_total'], dg_par['imno'], dg_par['td_max'], dg_par['mu_1'], dg_par['mu_2'], dg_par['mu_3'], dg_par['mu_4'], dg_par['dt_1'], dg_par['dt_2'], dg_par['dt_3'], dg_par['dt_4'], dg_par["mb"], dg_par["angsep_max"], dg_par["amplitude"]

