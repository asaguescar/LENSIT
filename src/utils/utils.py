# I want to combine the params and the targets
import glob
import pandas as pd
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import interp1d
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')


def combine_targets(globinput = '../sim_output/dset_ia_salt2_10000_?.pkl'):
    dsetfiles = glob.glob(globinput)
    dsets = []
    for dset_ in dsetfiles:
        df_ = pd.read_pickle(dset_)
        dsets.append(df_['targets'])
    return pd.concat(dsets, ignore_index=True)

def combine_target_and_lcparams(globinput = '../sim_output/dset_ia_salt2_10000_?.pkl'):
    dsetfiles = glob.glob(globinput)
    dsets = []
    for dset_ in dsetfiles:
        try:
            df_ = pd.read_pickle(dset_)
            lcpar_ = pd.read_pickle(dset_[:-4]+'_lcparams.pkl')
            try:
                fitpar_ = pd.read_pickle(dset_[:-4]+'_fittedparams.pkl')
                dfcomb_ = pd.concat([df_['targets'], lcpar_, fitpar_], axis=1)
            except:
                dfcomb_ = pd.concat([df_['targets'], lcpar_], axis=1)
            dfcomb_ = dfcomb_.loc[:, ~dfcomb_.columns.duplicated()] # to eliminate duplicated columns
            dsets.append(dfcomb_)
        except:
            continue
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

## find allowed parameter space by fluxratios constraint
##------------------------------------------------------

def doble_priors(resolved_epoch, fname, mu_1=[5, 15], mu_2=[1, 10], dt_1=[-15, 15], dt_2=[-15, 15],
                 t0=[59808.672 - 0.191, 59808.672 + 0.191], x1=[-2, 2],
                 c=[-.5, .5], n=10000, zs=0.3544, MWebv=0.08,
                 flux_q1=0.5, flux_q2=0.5, mu_total=25):
    '''
    resolved_epoch: list
    fname: txt file
    '''

    samples_name = ['mu_1', 'mu_2',
                    'dt_1', 'dt_2',
                    't0', 'x1', 'c']

    dust = sncosmo.CCM89Dust()
    image_1 = sncosmo.Model('salt2-extended')
    image_2 = sncosmo.Model('salt2-extended')
    model = sntd.unresolvedMISN([image_1, image_2])
    model.add_effect(dust, 'MW', 'obs')

    model.set(z=zs, MWebv=MWebv)

    samples_prior = []

    while len(samples_prior) < 2:
        print(len(samples_prior))
        q1 = float(np.random.uniform(v - .1, flux_q1 + .1, size=1))
        q2 = 1 - q1
        # print(q4,  q1 , q2 , q3)

        sample_ = np.array([float(mu_total * q1),
                            float(mu_total * q2),
                            np.random.uniform(dt_1[0], dt_1[1], 1)[0],
                            np.random.uniform(dt_2[0], dt_2[1], 1)[0],
                            np.random.uniform(t0[0], t0[1], 1)[0],
                            np.random.uniform(x1[0], x1[1], 1)[0],
                            np.random.uniform(c[0], c[1], 1)[0]
                            ])
        print(sample_)

        mu_1_, mu_2_, dt_1_, dt_2_, t0_, x1_, c_ = sample_

        model.set_delays([dt_1_, dt_2_], model.parameters[1])
        model.set_magnifications([mu_1_, mu_2_], model.parameters[2])

        model.set(t0=t0_, x1=x1_, c=c_)

        # ratios: q1,q2
        # epoch for the flux ratio constraint
        f1 = model.model_list[0].flux(t_keck, resolved_epoch)
        f2 = model.model_list[1].flux(t_keck, resolved_epoch)
        q1_ = float(f1 / f1)
        q2_ = float(f2 / f1)

        q1_, q2_ = np.array([q1_, q2_]) / np.sum([q1_, q2_])

        # print( q1_keck, q2_keck, q3_keck, q4_keck)
        mu_total_ = np.sum([mu_1_, mu_2_])
        # print(np.round(mu_total_),np.round(mu_total))

        # print( np.round(mu_total_)==np.round(mu_total) , keck_q1-0.013<q1_keck<keck_q1+0.013  ,  keck_q2-0.005<q2_keck<keck_q2+0.005  ,  keck_q3-0.011<q3_keck<keck_q3+0.011  ,  keck_q4-0.005<q4_keck<keck_q4+0.005 )

        if np.round(mu_total_) == np.round(
                mu_total) and flux_q1 - 0.01 < q1_ < flux_q1 + 0.01 and flux_q2 - 0.01 < q2_ < flux_q2 + 0.01:
            samples_prior.append([mu_1_, mu_2_, dt_1_, dt_2_, t0_, x1_, c_])
        else:
            continue

    samples_prior = np.array(samples_prior)

    file_1 = open(fname, 'wb')

    np.savetxt(file_1, samples_keck)


def targets_21mag_(lc):
    mag = 21
    peakmag_mask = (lc.g_modelpeak < mag) | (lc.r_modelpeak < mag) | (lc.i_modelpeak < mag)
    return peakmag_mask


def detection_criteria(lc):
    '''
    For detection ccriteria we require more than 5 data point with SNR more than 5 around peak: meaning between -10 and +20 days in any band.
    We also want this detections to happened at least during 5 days.
    We also want them to look brighter than a 1a SN. mb<-19.5 from the salt2 fit
    '''
    ndetaroundpeak_mask = (lc.g_npoints_aroundpeak + lc.r_npoints_aroundpeak + lc.i_npoints_aroundpeak) > 5
    mask = ndetaroundpeak_mask
    print('ndetaroundpeak_mask', sum(mask))
    dt_5s_mask = abs(lc.firstdet - lc.lastdet) > 5
    mask = mask & dt_5s_mask
    print('dt_5s_mask', sum(mask))
    mag = 21
    peakmag_mask = (lc.g_modelpeak < mag) | (lc.r_modelpeak < mag) | (lc.i_modelpeak < mag)
    mask = mask & peakmag_mask
    print('peakmag_mask<21', sum(mask))
    mb_fit_mask = lc.mb_fit < -19.5  # We assume here the mb_fit with the true zlens. Then we explore the uncertaintiesand potential misses from pherr
    mask = mask & mb_fit_mask
    print('mb_fit_mask', sum(mask))

    return mask


import numpy as np
from scipy.signal import savgol_filter


def find_turning_points(data_x, data_y, data_err, window_size, poly_order):
    # Apply Savitzky-Golay filter to smooth the data
    smoothed_y = savgol_filter(data_y, window_size, poly_order, mode='nearest')

    # Estimate the first derivative using weighted finite differences
    weights = 1.0 / (data_err ** 2)
    dx = np.gradient(data_x)
    dy = np.gradient(smoothed_y)
    derivative = np.gradient(dy / dx, dx) / weights

    # derivative = dy / dx
    derivative[abs(derivative) < .1] = 0
    print(derivative)
    print(np.sign(derivative))
    print(np.diff(np.sign(derivative)))
    print(np.where(np.diff(np.sign(derivative)) == -2)[0] + 1)

    # Find the indices where the derivative changes sign
    sign_changes = np.where(np.diff(np.sign(derivative)) == -2)[0] + 1

    # Get the turning points as (x, y) coordinates
    turning_points = [(data_x[idx], data_y[idx]) for idx in sign_changes]

    return turning_points


import numpy as np
from scipy.optimize import curve_fit


def polynomial_func(x, *coeffs):
    """
    Custom polynomial function to fit the data.
    """
    return np.polyval(coeffs, x)


def fit_polynomial(data_x, data_y, degree, data_err):
    # Initial guess for polynomial coefficients
    init_coeffs = np.zeros(degree + 1)

    # Fit the polynomial curve to the data with uncertainties
    popt, pcov = curve_fit(polynomial_func, data_x, data_y, p0=init_coeffs, sigma=data_err)

    # Calculate the number of peaks
    poly_coeffs = np.polyder(popt, m=1)
    poly_roots = np.roots(poly_coeffs)
    num_peaks = 0

    for root in poly_roots:
        if np.polyval(poly_coeffs, root - 1e-6) > 0 and np.polyval(poly_coeffs, root + 1e-6) < 0:
            num_peaks += 1

    return popt, num_peaks


def plot_lcs_band(dset_, idx, tcut, band='ztfg', color='green', alpha=1):
    model = dset_.targets.template.sncosmo_model
    model = None

    for i in idx:
        # model.set(**dset_.targets.data.loc[i][['z','t0', 'mu_1', 'mu_2', 'mu_3', 'mu_4', 'amplitude']])
        fig, ax = plt.subplots(1, 1)  # , figsize=(5,5))
        lc = dset_.data.loc[i]
        t0 = dset_.targets.data.loc[i].t0
        t0 = 0

        all_snr = lc['flux'] / lc['fluxerr']

        all_time = lc['time'] - t0
        time = {}
        flux = {}
        fluxerr = {}
        mag = {}
        magerr = {}
        snr = {}
        mlim = {}
        mask = lc['band'] == band
        time[band] = lc[mask]['time'] - t0
        flux[band] = lc[mask]['flux']
        fluxerr[band] = lc[mask]['fluxerr']
        mag[band] = 25 - 2.5 * np.log10(flux[band])
        mlim[band] = 25 - 2.5 * np.log10(5 * fluxerr[band])
        snr[band] = flux[band] / fluxerr[band]
        magerr[band] = 1.083 / snr[band]

        all_time5s = all_time.values  # [all_snr>5]
        # print(all_time5s)
        t = np.arange(all_time5s[0] - 10, all_time5s[-1] + 10, .2)

        # t0 = lc['time'][all_snr>5][lc['flux'][all_snr>5]==np.max(lc['flux'][all_snr>5])]

        if model != None:
            ml = model.bandmag(band, 'ab', t)
            ax.plot(t, ml, label='model')

        mask = snr[band] > 5
        ax.errorbar(time[band][mask], mag[band][mask], yerr=magerr[band][mask], linestyle='solid', color=color,
                    marker='.', alpha=alpha)
        # mask = (snr[band]>0) & (time[band]<tcut)
        # ax.errorbar(time[band][mask], mag[band][mask], yerr=magerr[band][mask], linestyle='solid', color=color, marker='.', alpha=alpha)
        # ax.scatter(time[band][mask], mag[band][mask], edgecolors='black', marker='o', zorder=10)
        mask = snr[band] <= 5
        ax.errorbar(time[band][mask], mlim[band][mask], linestyle='none', marker='v', alpha=.1)

        # ax.axvline(lcf.meta['t0'][i])

        ax.set_title(str(i) + ' z:' + str(dset_.targets.data.loc[i].z) + ' zl:' + str(
            dset_.targets.data.loc[i].zlens) + ' mag:' + str(dset_.targets.data.loc[i].magobslensed))
        # ax.set_title(str(i)+' zs:'+'{:.2f}'.format(lcf.meta['z'][i])+' zl:'+'{:.2f}'.format(lcf.meta['lensz'][i])+' dt:'+'{:.2f}'.format(lcf.meta['dt_max'][i])+' x0:'+'{:.2e}'.format(lcf.meta['x0'][i])+' x1:'+'{:.2f}'.format(lcf.meta['x1'][i])+' c:'+'{:.2f}'.format(lcf.meta['c'][i])+' mu:'+'{:.2f}'.format(lcf.meta['mu_total'][i]))
        # ax.set_ylim(22.5,18)
        # ax.set_xlim(t[0],t[-1])
        # ax.legend(loc=0)

        ax.invert_yaxis()

        ax.set_xlabel('epoch (days)')
        ax.set_ylabel('apparent magnitude')

        # ax.set_yticks(np.arange(20,25))

        fig.subplots_adjust(hspace=0)
        fig.show()
        # plt.savefig('lcs_v2.pdf')


import statsmodels.api as sm

path_lc = 'output_lc_rate_magcut/'
path_dset = 'output_dset_rate_magcut/'


def consistent_with_flat2(time_points, measurements, uncertainties):
    from sklearn.preprocessing import PolynomialFeatures
    # Choose the degree of the polynomial (e.g., quadratic)
    degree = 2

    # Perform polynomial regression
    X = sm.add_constant(time_points)
    weights = 1.0 / uncertainties ** 2
    model = sm.WLS(measurements, X, weights=weights)

    # Generate polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    results = model.fit(X_poly)

    # Get p-value for the highest degree coefficient
    p_value_poly = results.pvalues[-1]

    # Test if polynomial coefficient is consistent with 0
    alpha = 0.05  # Significance level
    if p_value_poly < alpha:
        print("The polynomial coefficient is statistically different from 0.")
        return False
    else:
        print("The polynomial coefficient is not statistically different from 0.")
        return True


def consistent_with_flat(time_points, measurements, uncertainties):
    X = sm.add_constant(time_points)
    weights = 1.0 / uncertainties ** 2

    model = sm.WLS(measurements, X, weights=weights)
    results = model.fit()
    print(results.params)

    try:
        slope = results.params[1]  # Slope coefficient
        intercept = results.params[0]  # Intercept coefficient
        p_value = results.pvalues[1]  # p-value for the slope coefficient
    except:
        slope = results.params[0]  # Slope coefficient
        intercept = np.nan  # Intercept coefficient
        p_value = results.pvalues[0]  # p-value for the slope coefficient

    print("Slope:", slope)
    print("Intercept:", intercept)
    print("p-value:", p_value)

    # Calculate the chi-squared statistic
    residuals = measurements - results.predict(X)
    chi_squared = np.sum((residuals / uncertainties) ** 2)

    # Degrees of freedom (number of data points - number of parameters)
    df = len(time_points) - len(results.params)

    red_chi_squared = chi_squared / df

    print("Chi-squared:", chi_squared)
    print("Degrees of Freedom:", df)
    print("reduced Chi-squared:", red_chi_squared)

    # Plot the regression line
    regression_line = intercept + slope * time_points
    plt.plot(time_points, regression_line, label='Regression', color='red')

    # Significance level
    alpha = 0.05  # Significance level
    alpha_redchi = 2  # Significance level
    if p_value < alpha and red_chi_squared >= alpha_redchi:
        print("Significantly nozero slope")
        return False
    elif p_value >= alpha and red_chi_squared < alpha_redchi:
        print("Significantly Flat")
        return True
    elif p_value >= alpha and red_chi_squared >= alpha_redchi and df > 0:
        print("Probably not flat")
        return False
    else:
        print("not enough data")
        return np.nan


def flat_test(lc=None, outputname='lc_Ia_params_newfits.pkl', targets=0, data=0):
    lc['flat_ztfg'] = np.ones(len(lc)) * np.nan
    lc['flat_ztfr'] = np.ones(len(lc)) * np.nan
    lc['flat_ztfi'] = np.ones(len(lc)) * np.nan

    lc_ = lc  # [lc.chisqlens==999]

    plt.rcParams["figure.figsize"] = (8, 4)

    doFit = True

    n = 0
    for index, index_dset in lc_[['index', 'index_dset']].values[:]:
        # for index, index_dset in data[['index', 'index_dset']].values:
        print(index, index_dset)
        targets_ = targets[targets['index_dset'] == index_dset]
        targets_.index = targets_['index']
        # print(targets_)
        data_ = data[data['index_dset'] == index_dset]
        data_.index = data_['index']
        # fig, ax = plot_lcs(ind=index, targets=targets_, data=data_, ind_snmodel='hsiao')
        # fig, ax  = plt.subplots()
        # if lcpar1a[(lcpar1a['index_dset']==index_dset)&(lcpar1a['index']==index)].chisqlens.values==9999:
        # if lc_[(lc_['index_dset']==index_dset)&(lc_['index']==index)].chisqlens.values>=990:

        data_ = data[(data['index_dset'] == index_dset) & (data['index'] == index) & (data['snr'] > 5)]
        data_table = Table.from_pandas(data_[['time', 'zp', 'zpsys', 'band', 'flux', 'fluxerr']])
        if len(data_table) == 0:
            continue

        plt.figure()
        for band in np.unique(data.band):

            print('\n--', band)
            mask_ = (data_.snr > 5) & (data_.band == band)
            time_points = data_.time[mask_]
            measurements = data_.mag[mask_]
            uncertainties = data_.magerr[mask_]
            if len(time_points) > 0:
                flat = consistent_with_flat(time_points, measurements, uncertainties)
                print(flat)
                index__ = lc[(lc['index_dset'] == index_dset) & (lc['index'] == index)].index
                lc['flat_' + band].loc[index__] = flat
                lc.to_pickle(path_lc + outputname)

            plt.errorbar(time_points, measurements, yerr=uncertainties, marker='o', linestyle='none', label=band)

        plt.legend(loc=0)
        plt.gca().invert_yaxis()
        plt.show()


def run_flat_test(sntype):
    print('---- ', sntype, ' ----')
    print('------------\n')

    lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params_newfits.pkl')
    targets = pd.read_csv(path_dset + 'dsetTargets_' + sntype + '_params.csv')
    data = pd.read_csv(path_dset + 'dsetData_' + sntype + '_lcs.csv')

    flat_test(lc=lc, outputname='lc_' + sntype + '_params_newfits.pkl', targets=targets, data=data)


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import sncosmo

#from get_lcs_params import fit_lc
from astropy.table import Table
# from ztf_sims import call_ztf_survey, get_lcs, get_lcs_det, plot_lcs
import warnings

warnings.filterwarnings("ignore")


def refits(lc=None, outputname='lc_Ia_params_newfits.pkl', modelcov=True, mwebv=None, targets=0, data=0):
    lc_ = lc[lc.chisqlens == 999]

    plt.rcParams["figure.figsize"] = (8, 4)

    doFit = True

    n = 0
    for index, index_dset in lc_[['index', 'index_dset']].values[:]:
        # for index, index_dset in data[['index', 'index_dset']].values:
        print(index, index_dset)
        targets_ = targets[targets['index_dset'] == index_dset]
        targets_.index = targets_['index']
        # print(targets_)
        data_ = data[data['index_dset'] == index_dset]
        data_.index = data_['index']
        # fig, ax = plot_lcs(ind=index, targets=targets_, data=data_, ind_snmodel='hsiao')
        # fig, ax  = plt.subplots()
        # if lcpar1a[(lcpar1a['index_dset']==index_dset)&(lcpar1a['index']==index)].chisqlens.values==9999:
        # if lc_[(lc_['index_dset']==index_dset)&(lc_['index']==index)].chisqlens.values>=990:

        print(lc_[(lc_['index_dset'] == index_dset) & (lc_['index'] == index)].zlens,
              lc_[(lc_['index_dset'] == index_dset) & (lc_['index'] == index)].mwebv)

        data_ = data[(data['index_dset'] == index_dset) & (data['index'] == index) & (data['snr'] > 5)]
        data_table = Table.from_pandas(data_[['time', 'zp', 'zpsys', 'band', 'flux', 'fluxerr']])
        if len(data_table) == 0:
            continue
        try:
            if doFit:
                if mwebv != 0:
                    mwebv = lc_[(lc_['index_dset'] == index_dset) & (lc_['index'] == index)].mwebv
                res, bft = fit_lc(data_table,
                                  redshift=lc_[(lc_['index_dset'] == index_dset) & (lc_['index'] == index)].zlens,
                                  mwebv=mwebv,
                                  modelcov=modelcov)
                _ = sncosmo.plot_lc(data_table, model=bft, errors=res.errors)
                print(res, bft)
                index__ = lc[(lc['index_dset'] == index_dset) & (lc['index'] == index)].index
                lc['t0lens'].loc[index__] = bft['t0']
                lc['x0lens'].loc[index__] = bft['x0']
                lc['x1lens'].loc[index__] = bft['x1']
                lc['clens'].loc[index__] = bft['c']
                lc['x1lens'].loc[index__] = bft['x1']
                lc['chisqlens'].loc[index__] = res.chisq
                lc['mb_fit'].loc[index__] = float(bft.source_peakabsmag('bessellb', 'ab'))
                lc.to_pickle(path_lc + outputname)
                n += 1
                print(n, ' fitted')
                print('\n---- ', sum(lc.chisqlens == 999), sum(lc.chisqlens != 999), len(lc), ' ----\n')
        except:
            print('\n---- No fit ----\n')
            _ = sncosmo.plot_lc(data_table)
        plt.show()
        print(lc[(lc['index_dset'] == index_dset) & (lc['index'] == index)][
                  ['num_peaks_g', 'num_peaks_r', 'num_peaks_i', 'zlens', 'zlens_phzerr', 'mwebv', 'chisqlens',
                   'mb_fit']])

        print('\n---- ', sum(lc.chisqlens == 999), sum(lc.chisqlens != 999), len(lc), ' ----\n')


def refits_phzerr(lc=None, outputname='lc_Ia_params_newfits.pkl', modelcov=True, mwebv=None, targets=0, data=0):
    lc_ = lc[lc.deltazl_chisqlens == 999]

    plt.rcParams["figure.figsize"] = (8, 4)

    doFit = True

    n = 0
    for index, index_dset in lc_[['index', 'index_dset']].values[:]:
        # for index, index_dset in data[['index', 'index_dset']].values:
        print(index, index_dset)
        targets_ = targets[targets['index_dset'] == index_dset]
        targets_.index = targets_['index']
        # print(targets_)
        data_ = data[data['index_dset'] == index_dset]
        data_.index = data_['index']
        # fig, ax = plot_lcs(ind=index, targets=targets_, data=data_, ind_snmodel='hsiao')
        # fig, ax  = plt.subplots()
        # if lcpar1a[(lcpar1a['index_dset']==index_dset)&(lcpar1a['index']==index)].chisqlens.values==9999:
        # if lc_[(lc_['index_dset']==index_dset)&(lc_['index']==index)].chisqlens.values>=990:

        print(lc_[(lc_['index_dset'] == index_dset) & (lc_['index'] == index)].zlens_phzerr,
              lc_[(lc_['index_dset'] == index_dset) & (lc_['index'] == index)].mwebv)

        data_ = data[(data['index_dset'] == index_dset) & (data['index'] == index) & (data['snr'] > 5)]
        data_table = Table.from_pandas(data_[['time', 'zp', 'zpsys', 'band', 'flux', 'fluxerr']])
        try:
            if doFit:
                if mwebv != 0:
                    mwebv = lc_[(lc_['index_dset'] == index_dset) & (lc_['index'] == index)].mwebv
                res, bft = fit_lc(data_table,
                                  redshift=lc_[
                                      (lc_['index_dset'] == index_dset) & (lc_['index'] == index)].zlens_phzerr,
                                  mwebv=mwebv,
                                  modelcov=modelcov)
                _ = sncosmo.plot_lc(data_table, model=bft, errors=res.errors)
                print(res, bft)
                index__ = lc[(lc['index_dset'] == index_dset) & (lc['index'] == index)].index
                lc['deltazl_t0lens'].loc[index__] = bft['t0']
                lc['deltazl_x0lens'].loc[index__] = bft['x0']
                lc['deltazl_x1lens'].loc[index__] = bft['x1']
                lc['deltazl_clens'].loc[index__] = bft['c']
                lc['deltazl_x1lens'].loc[index__] = bft['x1']
                lc['deltazl_chisqlens'].loc[index__] = res.chisq
                lc['deltazl_mb_fit'].loc[index__] = float(bft.source_peakabsmag('bessellb', 'ab'))
                lc.to_pickle(path_lc + outputname)
                n += 1
                print(n, ' fitted')
                print('\n---- ', sum(lc.deltazl_chisqlens == 999), sum(lc.deltazl_chisqlens != 999), len(lc), ' ----\n')
        except:
            print('\n---- No fit ----\n')
            _ = sncosmo.plot_lc(data_table)
        plt.show()
        print(lc[(lc['index_dset'] == index_dset) & (lc['index'] == index)][
                  ['num_peaks_g', 'num_peaks_r', 'num_peaks_i', 'zlens', 'zlens_phzerr', 'chisqlens',
                   'deltazl_chisqlens', 'mb_fit', 'deltazl_mb_fit']])

        print('\n---- ', sum(lc.chisqlens == 999), sum(lc.chisqlens != 999), len(lc), ' ----\n')
        print('\n---- ', sum(lc.deltazl_chisqlens == 999), sum(lc.deltazl_chisqlens != 999), len(lc), ' ----\n')


def run_refits(sntype):
    print('---- ', sntype, ' ----')
    print('------------\n')

    path_lc = 'output_lc_rate_magcut/'
    path_dset = 'output_dset_rate_magcut/'
    lc0 = pd.read_pickle(path_lc + 'lc_' + sntype + '_params.pkl')
    try:
        lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params_newfits.pkl')
    except:
        lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params.pkl')
    targets = pd.read_csv(path_dset + 'dsetTargets_' + sntype + '_params.csv')
    data = pd.read_csv(path_dset + 'dsetData_' + sntype + '_lcs.csv')

    print(sum(lc0.chisqlens == 999), sum(lc0.chisqlens != 999), len(lc0))
    print(sum(lc.chisqlens == 999), sum(lc.chisqlens != 999), len(lc))
    print()

    refits(lc=lc, outputname='lc_' + sntype + '_params_newfits.pkl', modelcov=True, targets=targets, data=data)
    try:
        lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params_newfits.pkl')
    except:
        lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params.pkl')
    refits(lc=lc, outputname='lc_' + sntype + '_params_newfits.pkl', modelcov=False, targets=targets, data=data)
    try:
        lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params_newfits.pkl')
    except:
        lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params.pkl')
    refits(lc=lc, outputname='lc_' + sntype + '_params_newfits.pkl', modelcov=True, mwebv=0, targets=targets, data=data)

    try:
        lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params_newfits.pkl')
    except:
        lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params.pkl')
    refits(lc=lc, outputname='lc_' + sntype + '_params_newfits.pkl', modelcov=False, mwebv=0, targets=targets,
           data=data)

    lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params_newfits.pkl')
    print()
    print(sum(lc0.deltazl_chisqlens == 999), sum(lc0.deltazl_chisqlens != 999), len(lc0))
    print(sum(lc.deltazl_chisqlens == 999), sum(lc.deltazl_chisqlens != 999), len(lc))
    print()
    refits_phzerr(lc=lc, outputname='lc_' + sntype + '_params_newfits.pkl', modelcov=True, targets=targets, data=data)
    lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params_newfits.pkl')
    refits_phzerr(lc=lc, outputname='lc_' + sntype + '_params_newfits.pkl', modelcov=False, targets=targets, data=data)
    lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params_newfits.pkl')
    refits_phzerr(lc=lc, outputname='lc_' + sntype + '_params_newfits.pkl', modelcov=True, mwebv=0, targets=targets,
                  data=data)
    lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params_newfits.pkl')
    refits_phzerr(lc=lc, outputname='lc_' + sntype + '_params_newfits.pkl', modelcov=False, mwebv=0, targets=targets,
                  data=data)
    lc = pd.read_pickle(path_lc + 'lc_' + sntype + '_params_newfits.pkl')


def plot_lens_ztf(
        numPix=10  # Number of pixels
        , deltaPix=1  # pixel size in angular units (1 in ZTF)
        , exp_time=30  # seconds
        , sigma_bkg=None  # background noise
        , fwhm=2  # arcsecond (2" for ZTF)
        , ellipticity=.5
        , theta_ein=.5
        , gamma=0.5
        , z_lens=.37
        , z_source=.7
        , zero_point=27.79  # 27
        , limiting_magnitude=24  # 29
        , sky_brightness=15.48  # 15
        , num_exposures=1
        , source_x=0.01
        , source_y=0.05
        , x_image=[-0.66330631, 0.94138169, 0.72518817, -0.50540839]
        , y_image=[0.84259064, -0.40711448, 0.66799182, -0.7405962]
        , app_mag=23  # apparent magnitude (without magnification)
        , macro_mag=[5.23, 6.76, 5.8, 3.39]
        , ccd_gain=2.3
        , read_noise=10
        , psf_type='GAUSSIAN'
        , truncation=5
        , cosmo=cosmo
        , savefig=False
        , filename=None
        , cmap='bone'
        , app_mag_lens=99
        , ax=None):
    import matplotlib.pyplot as plt
    import lenstronomy.Util.simulation_util as sim_util
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.Util import param_util
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    import lenstronomy.SimulationAPI.observation_api as observation_api
    from lenstronomy.PointSource.point_source import PointSource
    from lenstronomy.Plots.plot_util import coordinate_arrows, scale_bar

    obs_api = observation_api.SingleBand(pixel_scale=deltaPix, exposure_time=exp_time,
                                         magnitude_zero_point=zero_point, read_noise=read_noise, ccd_gain=ccd_gain,
                                         sky_brightness=sky_brightness, seeing=fwhm,
                                         num_exposures=num_exposures, psf_type=psf_type,
                                         kernel_point_source=None, truncation=truncation, data_count_unit='e-',
                                         background_noise=None)

    sigma_bkg = obs_api.background_noise
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
    data_class = ImageData(**kwargs_data)

    kwargs_psf = {'psf_type': psf_type, 'pixel_size': deltaPix, 'fwhm': fwhm}
    psf_class = PSF(**kwargs_psf)

    ## LENS
    lens_model_list = ['SIE', 'SHEAR']
    # SIE
    phi_lens = np.random.uniform(0, 2 * np.pi)
    q_lens = 1 - ellipticity
    e1_lens, e2_lens = param_util.phi_q2_ellipticity(phi_lens, q_lens)
    kwargs_sie = {'theta_E': theta_ein,
                  'e1': e1_lens, 'e2': e2_lens,
                  'center_x': 0., 'center_y': 0.}
    # External shear
    gamma1, gamma2 = param_util.shear_polar2cartesian(phi=np.random.uniform(0, 2 * np.pi),
                                                      gamma=gamma)
    kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}
    kwargs_lens = [kwargs_sie, kwargs_shear]
    lens_model_class = LensModel(lens_model_list=lens_model_list, z_lens=z_lens, z_source=z_source, cosmo=cosmo)

    # Light model
    lens_light_model_list = ['SERSIC_ELLIPSE']
    phi_light = 68.11 / 180 * np.pi
    q_light = 0.661
    e1, e2 = param_util.phi_q2_ellipticity(phi_light, q_light)
    amp_lens = obs_api.magnitude2cps(app_mag_lens) * exp_time
    kwargs_sersic_lens = {'amp': amp_lens, 'R_sersic': 1, 'n_sersic': 3,
                          'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0}
    kwargs_lens_light = [kwargs_sersic_lens]
    lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

    point_source_list = ['LENSED_POSITION']
    point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[False])
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

    # No host light model in difference images
    source_model_list = ['SERSIC_ELLIPSE']
    phi_source, q_source = 0, 0.5
    e1, e2 = param_util.phi_q2_ellipticity(phi_source, q_source)
    kwargs_sersic_source = {'amp': 0, 'R_sersic': 0.1, 'n_sersic': 2, 'e1': e1, 'e2': e2,
                            'center_x': source_x, 'center_y': source_y}
    kwargs_source = [kwargs_sersic_source]

    app_mag -= 2.5 * np.log10(macro_mag)
    amp_ps = obs_api.magnitude2cps(app_mag) * exp_time
    kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image, 'point_amp': amp_ps}]

    ## SOURCE
    source_model_list = ['SERSIC_ELLIPSE']
    source_model_class = LightModel(light_model_list=source_model_list)

    imageModel = ImageModel(data_class, psf_class, lens_model_class,
                            source_model_class, lens_light_model_class,
                            point_source_class, kwargs_numerics=kwargs_numerics)
    image_sim = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)

    max_coordinate, min_coordinate = max(data_class.pixel_coordinates[0][0]), min(data_class.pixel_coordinates[0][0])
    size = max_coordinate - min_coordinate  # width of the image in units of arc seconds  CORRECT!

    # Show the results
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.matshow((image_sim), origin='lower', cmap=cmap,
               extent=[min_coordinate - min_coordinate, max_coordinate - min_coordinate,
                       min_coordinate - min_coordinate, max_coordinate - min_coordinate])
    ax.scatter(source_x - min_coordinate, source_y - min_coordinate, marker='x', color='red')
    # ax.scatter(x_image, y_image, marker='o', color='blue')
    macro_mag_s = macro_mag / np.min(macro_mag)
    ax.scatter(x_image - min_coordinate, y_image - min_coordinate, marker='o', color='green', s=macro_mag_s)
    scale_bar(ax, d=size, dist=1., text='1"', color='w', font_size=15, flipped=False)
    for imno in range(len(x_image)):
        ax.text(x_image[imno] - min_coordinate, y_image[imno] - min_coordinate, s=str(int(imno + 1)), color='green')
    ax.xaxis.set_ticks_position('bottom')
    if savefig:
        plt.savefig(filename, transparent=False, facecolor='white', bbox_inches='tight', dpi=250)


def plot_data(data, ax=None, color_dict={'ztfg': 'green', 'ztfr': 'red', 'ztfi': 'orange'}):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for b in set(data['band']):
        mask = (data['band'] == b) & (data['snr'] > 3)
        ax.errorbar(data['time'][mask].values, data['mag'][mask].values,
                    yerr=data['magerr'][mask].values, linestyle='none', marker='.', color=color_dict[b], label=b)
    for b in set(data['band']):
        mask = (data['band'] == b) & (data['snr'] <= 3)
        ax.errorbar(data['time'][mask].values, data['maglim'][mask].values,
                    linestyle='none', marker='v', color=color_dict[b], alpha=.5)

    ax.set_xlabel('time')
    ax.set_ylabel('mag')


def plot_lensed_lcs(model, bands=['ztfg', 'ztfr', 'ztfi'],
                    color_dict={'ztfg': 'green', 'ztfr': 'red', 'ztfi': 'orange'},
                    linestyle='solid',
                    ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for b in bands:
        dt_max = np.max([model['dt_1'], model['dt_2'], model['dt_3'], model['dt_4']])
        time = np.linspace(model.mintime() - dt_max * (1 + model['z']), model.maxtime() + dt_max * (1 + model['z']),
                           100)
        mag = model.bandmag(b, 'ab', time)
        ax.plot(time, mag, color=color_dict[b], linestyle=linestyle)


def plot_model_lcs(model, bands=['ztfg', 'ztfr', 'ztfi'],
                   color_dict={'ztfg': 'green', 'ztfr': 'red', 'ztfi': 'orange'},
                   linestyle='solid',
                   ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for b in bands:
        time = np.linspace(model.mintime(), model.maxtime(), 100)
        mag = model.bandmag(b, 'ab', time)
        ax.plot(time, mag, color=color_dict[b], linestyle=linestyle)


def plot_lens_and_lcs_ia(target=0, data=0, survey=0, outputname=''):
    def closest_value(target, values):
        return min(values, key=lambda x: abs(x - target))
    try:
        data_peak = data[data.time == target.obs_t_peak]
        ind = data_peak.index[0]
    except:
        closest = closest_value(target.obs_t_peak, data.time)
        data_peak = data[data.time == closest]
        ind = data_peak.index[0]
        print('Exact peak not found, take the closest time value')

    # ADD survey details:
    data_peak['programid'] = np.nan
    data_peak['maglim'] = np.nan
    data_peak['fwhm'] = np.nan
    data_peak['gain'] = np.nan
    data_peak['expid'] = np.nan
    data_peak['exptime'] = np.nan
    data_peak['qcomment'] = np.nan
    mask = (survey.data.rcid == data_peak.loc[ind].rcid + 1) & (survey.data.fieldid == data_peak.loc[ind].fieldid + 1)
    mask = mask & (survey.data.band == data_peak.loc[ind].band)
    closest = closest_value(data_peak.loc[ind].time, survey.data[mask].mjd)
    mask = mask & (survey.data.mjd == closest)
    while np.sum(mask)>1:
        print('np.sum(mask)>1:', np.sum(mask))
        mask.loc[mask.index[mask==True].tolist()[0]] = False
    data_peak['programid'].loc[ind] = survey.data[mask]['programid'].values
    data_peak['maglim'].loc[ind] = survey.data[mask]['maglim'].values
    data_peak['fwhm'].loc[ind] = survey.data[mask]['fwhm'].values
    data_peak['gain'].loc[ind] = survey.data[mask]['gain'].values
    data_peak['expid'].loc[ind] = survey.data[mask]['expid'].values
    data_peak['exptime'].loc[ind] = survey.data[mask]['exptime'].values
    data_peak['qcomment'].loc[ind] = survey.data[mask]['qcomment'].values

    from src.simulations.glsne_target import GLSNe
    import sncosmo
    source = GLSNe('salt2', nimages=4)
    dust = sncosmo.CCM89Dust()
    lensed_model = sncosmo.Model(source, effects=[dust, dust], effect_names=['MW', 'host'],
                                 effect_frames=['obs', 'rest'])

    lensed_model.set(**{'z': target.z,
                        't0': target.t0,
                        'dt_1': target.dt_1,
                        'mu_1': target.mu_1,
                        'dt_2': target.dt_2,
                        'mu_2': target.mu_2,
                        'dt_3': target.dt_3,
                        'mu_3': target.mu_3,
                        'dt_4': target.dt_4,
                        'mu_4': target.mu_4,
                        'x0': target.x0,
                        'x1': target.x1,
                        'c': target.c,
                        'MWebv': target.MWebv,
                        'hostebv': target.hostebv,
                        'hostr_v': target.hostr_v})

    ind_model_1 = sncosmo.Model('salt2', effects=[dust, dust], effect_names=['MW', 'host'],
                                effect_frames=['obs', 'rest'])
    ind_model_1.set(**{'z': target.z,
                       't0': target.t0 + target.dt_1 * (1 + target['z']),
                       'x0': target.x0 * target.mu_1,
                       'x1': target.x1,
                       'c': target.c,
                       'MWebv': target.MWebv,
                       'hostebv': target.hostebv,
                       'hostr_v': target.hostr_v})

    ind_model_2 = sncosmo.Model('salt2', effects=[dust, dust], effect_names=['MW', 'host'],
                                effect_frames=['obs', 'rest'])
    ind_model_2.set(**{'z': target.z,
                       't0': target.t0 + target.dt_2 * (1 + target['z']),
                       'x0': target.x0 * target.mu_2,
                       'x1': target.x1,
                       'c': target.c,
                       'MWebv': target.MWebv,
                       'hostebv': target.hostebv,
                       'hostr_v': target.hostr_v})

    ind_model_3 = sncosmo.Model('salt2', effects=[dust, dust], effect_names=['MW', 'host'],
                                effect_frames=['obs', 'rest'])
    ind_model_3.set(**{'z': target.z,
                       't0': target.t0 + target.dt_3 * (1 + target['z']),
                       'x0': target.x0 * target.mu_3,
                       'x1': target.x1,
                       'c': target.c,
                       'MWebv': target.MWebv,
                       'hostebv': target.hostebv,
                       'hostr_v': target.hostr_v})

    ind_model_4 = sncosmo.Model('salt2', effects=[dust, dust], effect_names=['MW', 'host'],
                                effect_frames=['obs', 'rest'])
    ind_model_4.set(**{'z': target.z,
                       't0': target.t0 + target.dt_4 * (1 + target['z']),
                       'x0': target.x0 * target.mu_4,
                       'x1': target.x1,
                       'c': target.c,
                       'MWebv': target.MWebv,
                       'hostebv': target.hostebv,
                       'hostr_v': target.hostr_v})

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    mu_i = [target['mu_' + str(i + 1)] for i in range(int(target.imno))]
    plot_lens_ztf(exp_time=data_peak.exptime.values,
                  fwhm=float(data_peak.fwhm.values),
                  ellipticity=target.ellipticity_values,
                  theta_ein=target.einstein_radius_values,
                  gamma=target.gamma,
                  z_lens=target.z_l,
                  z_source=target.z,
                  zero_point=data_peak.zp.values,
                  limiting_magnitude=data_peak.maglim.values,
                  sky_brightness=10 ** (0.4 * (data_peak.zp.values - data_peak.maglim.values)) / 5,
                  num_exposures=1,
                  source_x=target.source_x,
                  source_y=target.source_y,
                  x_image=target.x_image,
                  y_image=target.y_image,
                  app_mag=target.MB + cosmo.distmod(target.z).value,
                  macro_mag=mu_i,
                  ccd_gain=data_peak.gain.values,
                  ax=ax[0, 0])

    # g
    plot_data(data[data.band == 'ztfg'], ax=ax[0, 1])
    plot_lensed_lcs(lensed_model, bands=['ztfg'], linestyle='solid', ax=ax[0, 1])
    plot_model_lcs(ind_model_1, bands=['ztfg'], linestyle='dashed', ax=ax[0, 1])
    plot_model_lcs(ind_model_2, bands=['ztfg'], linestyle='dashed', ax=ax[0, 1])
    plot_model_lcs(ind_model_3, bands=['ztfg'], linestyle='dashed', ax=ax[0, 1])
    plot_model_lcs(ind_model_4, bands=['ztfg'], linestyle='dashed', ax=ax[0, 1])
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_title('ZTFg')

    # r
    plot_data(data[data.band == 'ztfr'], ax=ax[1, 0])
    plot_lensed_lcs(lensed_model, bands=['ztfr'], linestyle='solid', ax=ax[1, 0])
    plot_model_lcs(ind_model_1, bands=['ztfr'], linestyle='dashed', ax=ax[1, 0])
    plot_model_lcs(ind_model_2, bands=['ztfr'], linestyle='dashed', ax=ax[1, 0])
    plot_model_lcs(ind_model_3, bands=['ztfr'], linestyle='dashed', ax=ax[1, 0])
    plot_model_lcs(ind_model_4, bands=['ztfr'], linestyle='dashed', ax=ax[1, 0])
    ax[1, 0].invert_yaxis()
    ax[1, 0].set_title('ZTFr')

    # r
    plot_data(data[data.band == 'ztfi'], ax=ax[1, 1])
    plot_lensed_lcs(lensed_model, bands=['ztfi'], linestyle='solid', ax=ax[1, 1])
    plot_model_lcs(ind_model_1, bands=['ztfi'], linestyle='dashed', ax=ax[1, 1])
    plot_model_lcs(ind_model_2, bands=['ztfi'], linestyle='dashed', ax=ax[1, 1])
    plot_model_lcs(ind_model_3, bands=['ztfi'], linestyle='dashed', ax=ax[1, 1])
    plot_model_lcs(ind_model_4, bands=['ztfi'], linestyle='dashed', ax=ax[1, 1])
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_title('ZTFi')

    lensing_params_txt = '$\mu_T=' + '{:.1f}'.format(target.mu_total) + '$: '
    for imno in range(1, int(target.imno) + 1):
        lensing_params_txt += '  $\mu_' + str(imno) + '=' + '{:.1f}'.format(target['mu_' + str(imno)]) + '$ '

    lensing_params_txt += '\n$\Delta t_{max}=' + '{:.1f}'.format(target.td_max) + '$: '
    for imno in range(1, int(target.imno) + 1):
        lensing_params_txt += '  $\Delta t_' + str(imno) + '=' + '{:.1f}'.format(target['dt_' + str(imno)]) + '$ '

    lensing_params_txt += '\n$z_{source}=' + '{:.2f}'.format(target.z) + '$'
    lensing_params_txt += '  $z_{lens}=' + '{:.2f}'.format(target.z_l) + '$'

    ax[0, 0].set_title(lensing_params_txt)

    for ax_ in [ax[0, 1], ax[1, 0], ax[1, 1]]:
        ax_.axvline(data_peak.time.values, color='grey', alpha=.3)

    for ax_ in [ax[0, 1], ax[1, 0], ax[1, 1]]:
        dt_min = np.min([target['dt_1'], target['dt_2'], target['dt_3'], target['dt_4']])
        ax_.set_xlim(lensed_model.mintime() - dt_min * (1 + target['z']),
                     lensed_model.maxtime() + target.td_max * (1 + target['z']))

    for ax_ in [ax[0, 1], ax[1, 0], ax[1, 1]]:
        ax_.set_ylim(data.mag[data.snr > 3].max() + 1, data.mag[data.snr > 3].min() - 1)

    plt.tight_layout()
    plt.savefig(outputname)
    plt.show()
