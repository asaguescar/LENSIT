import numpy as np

def add_photoz_error(zl, rel_err=0.15):
    phzerr = zl * rel_err
    zl_phzerr = [np.random.normal(zl[i], phzerr[i]) for i in range(len(zl))]
    return zl_phzerr, phzerr


def time_modelpeak(mod):
    '''
    adding in targets file time of peak and the magnitudes at the true peak extracted from the model
    '''
    wave = np.linspace(mod.minwave(), mod.maxwave(), 100)
    i, j = mod.mintime(), mod.maxtime()
    n = int(j - i)
    time = np.linspace(i, j, n)

    t_peak = time[np.argmax(np.max(mod.flux(time, wave), axis=1))]
    g_truepeak = mod.bandmag('ztfg', 'ab', t_peak)
    r_truepeak = mod.bandmag('ztfr', 'ab', t_peak)
    i_truepeak = mod.bandmag('ztfi', 'ab', t_peak)

    return t_peak, g_truepeak, r_truepeak, i_truepeak


def modelpeak(mod, t_peak):
    '''
    adding in targets file time of peak and the magnitudes at the true peak extracted from the model
    '''

    g_truepeak = mod.bandmag('ztfg', 'ab', t_peak)
    r_truepeak = mod.bandmag('ztfr', 'ab', t_peak)
    i_truepeak = mod.bandmag('ztfi', 'ab', t_peak)

    return g_truepeak, r_truepeak, i_truepeak

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


def get_lcs_params(times_, mags_, magerrs_, bands_, snr_, doPlot=False, survey_bands=['ztfg', 'ztfr', 'ztfi'],
                   snr_threshold=5., degree=2, brightest_point=False, polynomial_peak=True):
    '''
    In this function we provide the time, magnitude, magnitude error, bandpassed and the signal to noise ratio of the lightcurve data points.
    As an output it gives the infered observer peak magnitude and the colors.
    '''
    out_ = {}

    # We first clean the low snr datapoints:
    times, mags, magerrs, bands, snr = times_[snr_ > snr_threshold], mags_[snr_ > snr_threshold], magerrs_[
        snr_ > snr_threshold], bands_[snr_ > snr_threshold], snr_[snr_ > snr_threshold]
    ind_sort = np.argsort(times)
    times, mags, magerrs, bands, snr = times[ind_sort], mags[ind_sort], magerrs[ind_sort], bands[ind_sort], snr[
        ind_sort]

    # We take the peak from the band with more data points.
    ii = np.argmax([sum(bands == bb) for bb in survey_bands])
    band_mask = bands == survey_bands[ii]
    # print(survey_bands[ii])

    try:
        # fit quadratic function
        params, covariance = curve_fit(quadratic, times[band_mask], mags[band_mask], sigma=magerrs[band_mask])
        # Create a polynomial function using the coefficients
        polynomial = np.poly1d(params)

        x_range = np.linspace(min(times[band_mask]), max(times[band_mask]), 100)
        y_fit = polynomial(x_range)

        m_peak_ = y_fit[np.argmin(y_fit)]
        t_peak = x_range[np.argmin(y_fit)]

        weighted_residuals = abs((mags[band_mask] - polynomial(times[band_mask])) / magerrs[band_mask])
        mean_weighted_residuals = np.mean(weighted_residuals)

        doSpline = False
        if abs(mean_weighted_residuals) > 5:
            m_peak_ = np.nanmin(mags[band_mask])  # brightest point
            t_peak = float(times[band_mask][np.where(mags[band_mask] == m_peak_)[0]])
            doSpline = True
    except:
        m_peak_ = np.nanmin(mags[band_mask])  # brightest point
        t_peak = float(times[band_mask][np.where(mags[band_mask] == m_peak_)[0]])

    if doPlot:
        plt.figure()
        plt.errorbar(times[band_mask], mags[band_mask], yerr=magerrs[band_mask])
        # plt.plot(x_range, y_fit)
        plt.show()

    out_['obs_t_peak'] = t_peak
    out_['firstdet'] = min(times)
    out_['lastdet'] = max(times)
    # Get magnitudes from each band using polynomial with the same time, so we can extract the colors.
    for bb in survey_bands:
        t = times[bands == bb]
        m = mags[bands == bb]
        merr = magerrs[bands == bb]
        sn = snr[bands == bb]

        if ('array' in str(type(m))) == True and sum(~np.isnan(m)) > 0:
            m_i = m[~np.isnan(m)][0]  # magnitude from the first detection
            m_f = m[~np.isnan(m)][-1]  # magnitude of the last detection

            t_i = t[~np.isnan(m)][0]  # Time of first detection
            t_f = t[~np.isnan(m)][-1]  # Time of last detection

            x_range = np.linspace(t_i, t_f, 100)
            # print(m_i, m_f, t_i, t_f)
            if doPlot:
                plt.figure()
                plt.errorbar(t, m, yerr=merr, fmt='o', color='b', label='Experimental Data')

            try:
                if len(t) > 3:  # If more than 3 points we do polynomial fit
                    params, covariance = curve_fit(quadratic, t, m, sigma=merr)
                    polynomial = np.poly1d(params)
                    y_fit = polynomial(x_range)
                    if doSpline:
                        ff = interp1d(t, m, bounds_error=False)  # magnitude function at any epoch
                    else:
                        ff = interp1d(x_range, y_fit, bounds_error=False)  # magnitude function at any epoch

                    params, covariance = curve_fit(quadratic, t, m + merr, sigma=merr)
                    polynomial = np.poly1d(params)
                    if doSpline:
                        ff_up = interp1d(t, m + merr, bounds_error=False)
                        y_fit_upp = ff_up(x_range)
                    else:
                        y_fit_upp = polynomial(x_range)
                    params, covariance = curve_fit(quadratic, t, m - merr, sigma=merr)
                    polynomial = np.poly1d(params)
                    if doSpline:
                        ff_low = interp1d(t, m - merr, bounds_error=False)
                        y_fit_low = ff_low(x_range)
                    else:
                        y_fit_low = polynomial(x_range)
                    y_fit = np.array([np.max([y_fit_upp[i], y_fit_low[i]]) - np.min([y_fit_upp[i], y_fit_low[i]]) for i in
                                      range(len(x_range))])
                    ffmerr = interp1d(x_range, y_fit / 2, bounds_error=False)  # Funcction for magnitude errors
                    if doPlot:
                        plt.plot(x_range, y_fit_upp, '--', label='Upp bound')
                        plt.plot(x_range, y_fit_low, ':', label='Low bound')
                        plt.plot(x_range, ff(x_range), 'r-', label='Fit')
                elif len(t) > 1:
                    ff = interp1d(t, m, bounds_error=False)
                    ffmerr = interp1d(t, merr, bounds_error=False)
                    if doPlot:
                        plt.plot(x_range, ff(x_range), 'r-', label='Fit')
                        plt.plot(x_range, ff(x_range) + ffmerr(x_range), '--', label='Upp bound')
                        plt.plot(x_range, ff(x_range) - ffmerr(x_range), ':', label='Low bound')
                else:
                    ff = interp1d([-20, 50], [np.nan, np.nan], bounds_error=False)
                    ffmerr = interp1d([-20, 50], [np.nan, np.nan], bounds_error=False)
            except:
                try:
                    ff = interp1d(t, m, bounds_error=False)
                    ffmerr = interp1d(t, merr, bounds_error=False)
                    if doPlot:
                        plt.plot(x_range, ff(x_range), 'r-', label='Fit')
                        plt.plot(x_range, ff(x_range) + ffmerr(x_range), '--', label='Upp bound')
                        plt.plot(x_range, ff(x_range) - ffmerr(x_range), ':', label='Low bound')
                except:
                    ff = interp1d([-20, 50], [np.nan, np.nan], bounds_error=False)
                    ffmerr = interp1d([-20, 50], [np.nan, np.nan], bounds_error=False)

            m_peak = ff(t_peak)

            if doPlot:
                plt.errorbar(t_peak, m_peak, yerr=ffmerr(t_peak), fmt='o', color='r', label='peak')

                plt.xlabel('t')
                plt.ylabel('m ' + bb)
                plt.legend()
                plt.show()

            out_[bb + '_npoints'] = np.array(sum(~np.isnan(m)))
            epoch = t - t_peak
            mask_peak = (epoch < 20) & (epoch > -10)
            out_[bb + '_npoints_aroundpeak'] = np.array(sum(~np.isnan(m[mask_peak])))
            out_[bb + '_rise_time'] = t_peak - t_i
            if out_[bb + '_rise_time'] > 1:
                out_[bb + '_rise_rate'] = (m_peak - m_i) / out_[bb + '_rise_time']
                if out_[bb + '_rise_rate'] > 0:
                    out_[bb + '_rise_rate'] = np.nan
            else:
                out_[bb + '_rise_rate'] = np.nan
                out_[bb + '_rise_time'] = np.nan
            out_[bb + '_decay_time'] = t_f - t_peak
            if out_[bb + '_decay_time'] > 1:
                out_[bb + '_decay_rate'] = (m_f - m_peak) / out_[bb + '_decay_time']
                if out_[bb + '_decay_rate'] < 0:
                    out_[bb + '_decay_rate'] = np.nan
            else:
                out_[bb + '_decay_rate'] = np.nan
                out_[bb + '_decay_time'] = np.nan
            out_[bb + '_peak_mag'] = m_peak
            out_[bb + '_peak_mag_minus7'] = ff(t_peak - 7)
            out_[bb + '_peak_mag_plus7'] = ff(t_peak + 7)
            out_[bb + '_peak_mag_minus14'] = ff(t_peak - 14)
            out_[bb + '_peak_mag_plus14'] = ff(t_peak + 14)
            out_[bb + '_peak_mag_err'] = ffmerr(t_peak)
            out_[bb + '_peak_mag_minus7_err'] = ffmerr(t_peak - 7)
            out_[bb + '_peak_mag_plus7_err'] = ffmerr(t_peak + 7)
            out_[bb + '_peak_mag_minus14_err'] = ffmerr(t_peak - 14)
            out_[bb + '_peak_mag_plus14_err'] = ffmerr(t_peak + 14)

            t_color = np.arange(-20, 50.1, 5) + t_peak
            out_[bb + '_peak_mag_epochs'] = ff(t_color)
            out_[bb + '_peak_magerr_epochs'] = ffmerr(t_color)
        else:
            out_[bb + '_npoints'] = sum(~np.isnan(m))
            epoch = t - t_peak
            mask_peak = (epoch < 20) & (epoch > -10)
            out_[bb + '_npoints_aroundpeak'] = np.array(sum(~np.isnan(m[mask_peak])))
            out_[bb + '_rise_time'] = np.nan
            out_[bb + '_rise_rate'] = np.nan
            out_[bb + '_decay_time'] = np.nan
            out_[bb + '_decay_rate'] = np.nan
            out_[bb + '_peak_mag'] = np.nan
            out_[bb + '_peak_mag_minus7'] = np.nan
            out_[bb + '_peak_mag_plus7'] = np.nan
            out_[bb + '_peak_mag_minus14'] = np.nan
            out_[bb + '_peak_mag_plus14'] = np.nan
            out_[bb + '_peak_mag_err'] = np.nan
            out_[bb + '_peak_mag_minus7_err'] = np.nan
            out_[bb + '_peak_mag_plus7_err'] = np.nan
            out_[bb + '_peak_mag_minus14_err'] = np.nan
            out_[bb + '_peak_mag_plus14_err'] = np.nan
            out_[bb + '_peak_mag_epochs'] = np.nan
            out_[bb + '_peak_magerr_epochs'] = np.nan
    # colors
    # out_['colors'] = {}
    bands_ = survey_bands
    for cc1 in range(len(bands_)):
        c1 = bands_[cc1]
        for cc2 in range(cc1 + 1, len(bands_)):
            c2 = bands_[cc2]
            out_[c1 + '-' + c2] = np.array(out_[c1 + '_peak_mag']) - np.array(out_[c2 + '_peak_mag'])
            out_[c1 + '-' + c2 + '_err'] = np.sqrt(
                np.array(out_[c1 + '_peak_mag_err']) ** 2 + np.array(out_[c2 + '_peak_mag_err']) ** 2)

            out_[c1 + '-' + c2 + '_minus7'] = np.array(out_[c1 + '_peak_mag_minus7']) - np.array(
                out_[c2 + '_peak_mag_minus7'])
            out_[c1 + '-' + c2 + '_minus7_err'] = np.sqrt(
                np.array(out_[c1 + '_peak_mag_minus7_err']) ** 2 + np.array(out_[c2 + '_peak_mag_minus7_err']) ** 2)

            out_[c1 + '-' + c2 + '_plus7'] = np.array(out_[c1 + '_peak_mag_plus7']) - np.array(
                out_[c2 + '_peak_mag_plus7'])
            out_[c1 + '-' + c2 + '_plus7_err'] = np.sqrt(
                np.array(out_[c1 + '_peak_mag_plus7_err']) ** 2 + np.array(out_[c2 + '_peak_mag_plus7_err']) ** 2)

            out_[c1 + '-' + c2 + '_minus14'] = np.array(out_[c1 + '_peak_mag_minus14']) - np.array(
                out_[c2 + '_peak_mag_minus14'])
            out_[c1 + '-' + c2 + '_minus14_err'] = np.sqrt(
                np.array(out_[c1 + '_peak_mag_minus14_err']) ** 2 + np.array(out_[c2 + '_peak_mag_minus14_err']) ** 2)

            out_[c1 + '-' + c2 + '_plus14'] = np.array(out_[c1 + '_peak_mag_plus14']) - np.array(
                out_[c2 + '_peak_mag_plus14'])
            out_[c1 + '-' + c2 + '_plus14_err'] = np.sqrt(
                np.array(out_[c1 + '_peak_mag_plus14_err']) ** 2 + np.array(out_[c2 + '_peak_mag_plus14_err']) ** 2)

    if doPlot:
        plt.figure()
        # plt.errorbar(times, mags, yerr=magerrs, linestyle='none', marker='o')
        for b in bands_:
            plt.errorbar(times[bands == b], mags[bands == b], yerr=magerrs[bands == b], linestyle='none', marker='o',
                         label=b)
        # plt.scatter(times, mags, c=bands)
        plt.plot([t_peak], [m_peak_], marker='x', markersize=50, color='black')
        plt.gca().invert_yaxis()
        # plt.savefig()
        plt.legend(loc=0)
        plt.show()

    return out_


def quadratic(x, d, a, b, c):
    return d * x ** 3 + a * x ** 2 + b * x + c

