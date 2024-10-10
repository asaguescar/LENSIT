import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import interp1d
from scipy.integrate import quad
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')

class plotting:
    def corner_plot(self, df, params=['dmag', 'log_td_max', 'log_angsep_max'], 
                    param_labels=[r'$\Delta m$', r'$\log(\Delta t_{max})$', r'$\theta_{max}$'],
                    weights = None, levels=[1-.95,1- .68, 1], 
                    hue='imno', hue_order=[2.0,3.0,4.0], 
                    common_norm=False, aspect=1, despine=True, palette='bright',
                    legend_labels=['Doubles', 'Triplets', 'Quads'],
                    savefigname = ''):
        """
        Generate a corner plot for the given DataFrame showcasing relationships between multiple parameters.

        Parameters:
            df (DataFrame): The data frame containing the data.
            params (list): List of parameters to plot.
            param_labels (list): List of labels for the parameters.
            weights (array, optional): Weights for each point in the plot.
            levels (list): Contour levels to plot.
            hue (str): Column name for hue categorization.
            hue_order (list): Order of hues for categorical distinction.
            common_norm (bool): Whether to apply a common normalization across facets.
            aspect (float): Aspect ratio of each facet.
            despine (bool): Whether to despine the figures of their box frame.
            palette (str): Color palette to use.
            legend_labels (list): Labels for the legend.
            savefigname (str): Path to save the figure file.

        Returns:
            None
        """

        if hue is not None:
            cols = list(np.concatenate([params, [hue]]))
        else:
            cols = params
        g = sns.PairGrid(df[cols], hue=hue, hue_order=hue_order,
                 corner=True, aspect=aspect, despine=despine, palette=palette)
        # KDE plot on diagonal
        g.map_diag(sns.histplot, weights=weights, common_norm=common_norm, element='step', stat='proportion')
        # KDE contours on lower triangle
        g.map_lower(sns.kdeplot, levels=levels, weights=weights, common_norm=common_norm, 
                    fill=True, alpha=.75, cut=0)
        g.map_lower(sns.kdeplot, levels=levels, weights=weights, common_norm=common_norm, 
                    fill=False, alpha=1, cut=0)

        plt.legend(labels=legend_labels, loc=7)

        for i in range(len(params)):
            # xlabel 
            g.axes[len(params)-1, i].set_xlabel(param_labels[i])
            # ylabel 
            if i!=0:
                g.axes[i, 0].set_ylabel(param_labels[i])
    
        ## Customize layout 
        g.fig.subplots_adjust(top=0.95, wspace=0.05, hspace=0.05)
                                 
        #plt.tight_layout()
        plt.savefig( savefigname, bbox_inches='tight')
        plt.show()

    
    def plot_lens_ztf(self,
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
        """
        Simulate the lensed appearance of an astronomical object as it would appear in ZTF imaging data.

        Parameters:
            numPix (int): Number of pixels across the image dimension.
            deltaPix (float): Pixel scale (arcseconds per pixel).
            exp_time (int): Exposure time in seconds.
            sigma_bkg (float): Background noise level.
            fwhm (float): Full-width at half maximum (arcseconds) for the PSF.
            ellipticity (float): Ellipticity of the lens galaxy.
            theta_ein (float): Einstein radius in arcseconds.
            gamma (float): External shear strength.
            z_lens (float): Redshift of the lens.
            z_source (float): Redshift of the source.
            zero_point (float): Photometric zero point.
            limiting_magnitude (float): Limiting magnitude of the survey.
            sky_brightness (float): Sky brightness in magnitudes per square arcsecond.
            num_exposures (int): Number of exposures.
            source_x (float): Source position along x.
            source_y (float): Source position along y.
            x_image (list): x-coordinates of images.
            y_image (list): y-coordinates of images.
            app_mag (float): Apparent magnitude of the source.
            macro_mag (list): List of magnification factors for each image.
            ccd_gain (float): CCD gain (electrons per count).
            read_noise (float): Read noise (electrons).
            psf_type (str): Type of PSF model.
            truncation (float): Truncation radius for the PSF.
            cosmo: Cosmology used for distance calculations.
            savefig (bool): If True, save the figure.
            filename (str): Filename to save the figure.
            cmap (str): Colormap for the image.
            app_mag_lens (float): Apparent magnitude of the lens galaxy.
            ax (matplotlib axis): Axis to plot on, if provided.

        Returns:
            None
        """
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
    


def funcmax(df, i):
    """
    Calculate the maximum separation between images in a lens system.

    Parameters:
        df (DataFrame): DataFrame containing x and y image positions.
        i (int): Index of the lens system in the DataFrame.

    Returns:
        float: Maximum separation between any two images in arcseconds.
    """
    xs = df['x_image'].values[i]
    ys = df['y_image'].values[i]
    d = []
    for i in range(len(xs)):
        d.append( np.sqrt( (xs-xs[i])**2 + (ys-ys[i])**2 ) )
    d = np.array(d)
    return np.max(d)


def get_detections_around_peak(t_peak, time, mags, magerr, bands, snr, snr_threshold=5.):
    """
    Identify detections around the peak brightness within a specified range in time and binned per day.

    Parameters:
        t_peak (float): Time of peak brightness.
        time (array): Array of times for observations.
        mags (array): Magnitudes corresponding to the times.
        magerr (array): Errors associated with the magnitudes.
        bands (array): Bands in which the observations were made.
        snr (array): Signal-to-noise ratios of the observations.
        snr_threshold (float): Threshold for considering detections based on SNR.

    Returns:
        dict: A dictionary with keys as band names and values as the number of points around the peak.
    """
    out = {}
    for bb in np.unique(bands):
        mask = (snr>snr_threshold)&(bands==bb)
        t = time[mask]
        epoch = t - t_peak
        m = mags[mask]
        # bin epoch per day
        epoch_binned = np.unique(np.round(epoch[~np.isnan(m)], decimals=0))
        mask_peak = (epoch_binned <= 20) & (epoch_binned >= -10)
    
        out[bb + '_npoints_aroundpeak'] = np.array(sum(mask_peak))
        
    return out
    
def event_rate_ia(z):
    """
    Calculate the intrinsic event rate of Type Ia supernovae based on redshift, incorporating evolution and decay terms.

    Parameters:
        z (array): Array of redshifts.

    Returns:
        array: Calculated event rates for the given redshifts in Gpc-3
    """
    try:
        rate = np.zeros(len(z))
        rate[z<1] = 2.35e4 * (1+z[z<1])**1.5
        rate[z>=1] = 2.35e4 * (1+1)**1.5 / (1+1)**(-0.5) * (1+z[z>=1])**(-0.5)
    except:
        if z<1: rate = 2.35e4 * (1+z)**1.5
        elif z>=1: rate = 2.35e4 * (1+1)**1.5 / (1+1)**(-0.5) * (1+z)**(-0.5)  
    return rate
    
def comoving_rate_ia(z, omega_sky=4*np.pi):
    """
    Calculate the comoving rate of events across the sky up to a given redshift. Rsl(< zmax)

    Parameters:
        z (array): Array of redshifts to calculate up to.
        omega_sky (float): Total sky area covered by the survey in steradians.

    Returns:
        array: Comoving rates integrated up to each redshift.
    """
    from src.lenses import LensGalaxy
    lens = LensGalaxy()
    def integral_function(z):
        return cosmo.differential_comoving_volume(z).to('Gpc3/sr').value * event_rate_ia(z) / (1+z) * lens.psl_simpl(z)
    return omega_sky * np.array([quad(integral_function, 0, zmax)[0] for zmax in z])

def add_weight_ia(z, theta_ein, zmax=2, Rloc = 2.35e4, alpha= 1.):
    """
    Assign weights to different redshifts based on their lensing probabilities and volume elements.

    Parameters:
        z (array): Array of redshifts.
        theta_ein (float): Einstein radius in radians.
        zmax (float): Maximum redshift considered.
        Rloc (float): Local density rate.
        alpha (float): Scaling exponent for the rate density.

    Returns:
        array: Weighted values for each redshift interval.
    """
    weight = theta_ein**2

    z_bins = np.arange(.1, np.max(z)+.1, 0.01)
    for i in range(len(z_bins)-1):
        z_1, z_2 = z_bins[i], z_bins[i+1]
        rate_bin = (float(comoving_rate_ia([z_2])) -  float(comoving_rate_ia([z_1])) )

        mask = (z<=z_2) & (z>z_1)
        weight[mask] = weight[mask]/weight[mask].sum() * rate_bin

    return weight


def Rcc(z, k = 0.01, A=0.015, B=1.5, C=5.0, D=6.1):
    """
    Calculate the cosmic core-collapse supernova rate using parameters from Strolger et al. 2015.

    Parameters:
        z (array): Array of redshifts.
        k, A, B, C, D (float): Parameters defining the rate evolution.

    Returns:
        array: Rates of core-collapse supernovae per redshift.
    """
    phi = A * (1+z)**C /( ((1+z)/B)**D + 1)
    R = k* phi
    R0 = k* A * (1)**C /( ((1)/B)**D + 1)
    R0_Perley = 1.01e5
    return R / R0 * R0_Perley

def Rcc_MD(z, k = 0.0068, A=0.015, B=2.9, C=2.7, D=5.6):
    """
    Calculate the cosmic core-collapse supernova rate using parameters from Madau and Dickinson et al. 2014.

    Parameters:
        z (array): Array of redshifts.
        k, A, B, C, D (float): Parameters defining the rate evolution.

    Returns:
        array: Rates of core-collapse supernovae per redshift.
    """
    phi = A * (1+z)**C /( ((1+z)/B)**D + 1)
    R = k* phi
    R0 = k* A * (1)**C /( ((1)/B)**D + 1)
    R0_Perley = 1.01e5
    return R / R0 * R0_Perley


def event_rate_iip(z, relative_rate=.298):
    #return Rcc_MD(z) * relative_rate
    R_ = 5.52e4
    return  Rcc_MD(z) /Rcc_MD(0) * R_
    


def comoving_rate_iip(z, omega_sky=4*np.pi):
    '''Rsl(< zmax)
    omega_sky : sky area of the survey'''

    from src.lenses import LensGalaxy
    lens = LensGalaxy()

    def integral_function(z):
        return cosmo.differential_comoving_volume(z).to('Gpc3/sr').value * event_rate_iip(z) / (1+z) * lens.psl_simpl(z)
    return omega_sky * np.array([quad(integral_function, 0, zmax)[0] for zmax in z])


def add_weight_iip(z, theta_ein, zmax=2):
    weight = theta_ein**2

    z_bins = np.arange(.1, np.max(z)+.1, 0.01)
    for i in range(len(z_bins)-1):
        z_1, z_2 = z_bins[i], z_bins[i+1]
        rate_bin = (float(comoving_rate_iip([z_2])) -  float(comoving_rate_iip([z_1])) )

        mask = (z<=z_2) & (z>z_1)
        weight[mask] = weight[mask]/weight[mask].sum() * rate_bin
        
    return weight


def event_rate_iin(z, relative_rate=.1025):
    #return Rcc_MD(z) * relative_rate
    R_ = 5.05e3
    return Rcc_MD(z) /Rcc_MD(0) * R_
    


def comoving_rate_iin(z, omega_sky=4*np.pi):
    '''Rsl(< zmax)
    omega_sky : sky area of the survey'''

    from src.lenses import LensGalaxy
    lens = LensGalaxy()
    def integral_function(z):
        return cosmo.differential_comoving_volume(z).to('Gpc3/sr').value * event_rate_iin(z) / (1+z) * lens.psl_simpl(z)
    return omega_sky * np.array([quad(integral_function, 0, zmax)[0] for zmax in z])


def add_weight_iin(z, theta_ein, zmax=2):
    weight = theta_ein**2

    z_bins = np.arange(.1, np.max(z)+.1, 0.01)
    for i in range(len(z_bins)-1):
        z_1, z_2 = z_bins[i], z_bins[i+1]
        rate_bin = (float(comoving_rate_iin([z_2])) -  float(comoving_rate_iin([z_1])) )

        mask = (z<=z_2) & (z>z_1)
        weight[mask] = weight[mask]/weight[mask].sum() * rate_bin

    return weight

    
def event_rate_ibc(z, relative_rate=.1629):
    R_ = 3.33e4
    #return Rcc_MD(z) * relative_rate
    return Rcc_MD(z) /Rcc_MD(0) * R_


def comoving_rate_ibc(z, omega_sky=4*np.pi):
    '''Rsl(< zmax)
    omega_sky : sky area of the survey'''
    
    from src.lenses import LensGalaxy
    lens = LensGalaxy()
    def integral_function(z):
        return cosmo.differential_comoving_volume(z).to('Gpc3/sr').value * event_rate_ibc(z) / (1+z) * lens.psl_simpl(z)
    return omega_sky * np.array([quad(integral_function, 0, zmax)[0] for zmax in z])


def add_weight_ibc(z, theta_ein, zmax=2):
    weight = theta_ein**2

    z_bins = np.arange(.1, np.max(z)+.1, 0.01)
    for i in range(len(z_bins)-1):
        z_1, z_2 = z_bins[i], z_bins[i+1]
        rate_bin = (float(comoving_rate_ibc([z_2])) -  float(comoving_rate_ibc([z_1])) )

        mask = (z<=z_2) & (z>z_1)
        weight[mask] = weight[mask]/weight[mask].sum() * rate_bin

    return weight

# k-correction
import sncosmo
import numpy as np

def KBRcorr(z):
    salt2 = sncosmo.Model('salt2')
    fB = salt2.bandflux('bessellb', 0, zpsys='Vega')
    fr = salt2.bandflux('ztfr'   , 0, zpsys='ab'  )
    salt2.set(**{'z':z})
    fr_z = salt2.bandflux('ztfr'   , 0, zpsys='ab'  )
    KBr = -2.5*np.log10(fB/fr) + 2.5*np.log10(fB/fr_z)
    #print(fB, fr, fr_z, KBr)
    return KBr


def time_peak(data, bands=['ztfg', 'ztfr', 'ztfi']):

    return data.time.values[np.argmax(data[data.snr>5].flux.values)]
    


def obs_peakmag(t_peak, data, bands=['ztfg', 'ztfr', 'ztfi']):
    '''
    To get peak magnitude
    t_peak: in days
    data: pandas dataframe
    '''
    time_data_peak = {}
    mag_data_peak = {}
    for bb in bands:
        bb_mask = (data.band==bb)&(data.snr>=5)
        data_bb_mask = data[bb_mask]
        time_data_peak[bb] = data_bb_mask.time[abs(data_bb_mask.time - t_peak)<=3]
        mag_data_peak[bb] = np.nanmean(data_bb_mask.mags.loc[time_data_peak[bb].index])

    return mag_data_peak['ztfg'], mag_data_peak['ztfr'], mag_data_peak['ztfi']


    
def format_skysurvey_outputs(dsets_list, sntype='ia'):
    """
    Processes datasets from sky surveys to categorize supernova detection status and computes additional metrics.

    Parameters:
        dsets_list (list): List of datasets, each containing 'data' and 'targets' as keys.
        sntype (str): Type of supernova ('ia', 'iip', 'iin', 'ibc') to apply specific weights.

    Returns:
        dict: Dictionary containing four categories of supernova detection status:
              - not_observed: Targets not observed.
              - not_detected: Targets observed but not detected.
              - detectable: Targets detected with sufficient data points.
              - identifiable: Detected targets that can be confidently identified as supernovae based on their inferred magnitudes.

    Raises:
        ValueError: If the supernova type `sntype` is not recognized.
    """
    not_observed = {'targets': []}

    not_detected = {'data': [],
                    'targets': []}
    
    detectable = {'data': [],
                  'targets': []}
    
    if sntype=='ia':
        add_weight = add_weight_ia
    elif sntype=='iip':
        add_weight = add_weight_iip
    elif sntype=='iin':
        add_weight = add_weight_iin
    elif sntype=='ibc':
        add_weight = add_weight_ibc
    else:
        raise ValueError("sntype undefined")

    
    max_index = 0
    for i in range(len(dsets_list)):
        print(i+1,'/',len(dsets_list))
        dset_ = pd.read_pickle(dsets_list[i])
        data = dset_['data']
        targets = dset_['targets']
        targets['weight'] = add_weight(targets.z, targets.theta_ein, zmax=1.5)/len(dsets_list)
        targets['dmag'] = 2.5*np.log10(targets['mu_total'])
        
        #targets['angsep_max'] = [funcmax(targets, i) for i in range(len(targets['z']))]
    
        min_index = max_index + 1
        max_index = min_index + len(targets)
        new_index = np.arange(min_index, max_index, 1) # min_index, max_index inclusive
        old_index = targets.index.values
        targets.set_index(pd.Index(new_index), inplace=True)
    
        data['mags'] = data.zp.values - 2.5 * np.log10(data.flux.values)
        data['magerr'] = 2.5 / np.log(10) * (data.fluxerr.values / data.flux.values)
        data['snr'] = data.flux.values / data.fluxerr.values

        data = data[data['snr']>=3]
        
        data['snindex'] = np.nan

        targets['t_peak'] = time_peak(data)
    
        # We are only saving data from detected sources. ndet>5 epochs around peak
        targets['ztfg_npoints_aroundpeak'] = np.nan
        targets['ztfr_npoints_aroundpeak'] = np.nan
        targets['ztfi_npoints_aroundpeak'] = np.nan
        targets['npoints_aroundpeak'] = np.nan
        
        for ind in tqdm(targets[targets.ndet>=5].index):
            old_index_ = old_index[new_index==ind]
            data['snindex'].loc[old_index_] = ind 
            
            out = get_detections_around_peak(targets.loc[ind].t_peak, 
                                           data.loc[old_index_].time.values, 
                                           data.loc[old_index_].mags.values,
                                           data.loc[old_index_].magerr.values,
                                           data.loc[old_index_].band.values,
                                           data.loc[old_index_].snr.values, snr_threshold=5.)
            npoints_aroundpeak = 0
            for k in out.keys():
                targets[k].loc[ind]  = out[k]
                npoints_aroundpeak += out[k]
            targets['npoints_aroundpeak'].loc[ind] = npoints_aroundpeak
            
        not_observed['targets'].append( targets[np.isnan(targets.ndet)] )
        
        not_detected['targets'].append( targets[~np.isnan(targets.ndet) & (targets.npoints_aroundpeak<5)] )
        
        detectable['targets'].append( targets[targets.npoints_aroundpeak>=5] )
        detectable['data'].append( data[np.isin(data.snindex.values, targets[targets.npoints_aroundpeak>=5].index)] )
    
    not_observed['targets'] = pd.concat(not_observed['targets'])
    not_detected['targets'] = pd.concat(not_detected['targets'])
    detectable['targets'] = pd.concat(detectable['targets'])
    detectable['data'] = pd.concat(detectable['data'], ignore_index=True)


    # Add apparent peak magnitude
    
    snindex = detectable['targets'].index
    
    detectable['targets']['g_peak_mag'] = np.nan
    detectable['targets']['r_peak_mag'] = np.nan
    detectable['targets']['i_peak_mag'] = np.nan
    detectable['targets']['min_peak_mag'] = np.nan
    detectable['targets']['KBr'] = np.nan
    detectable['targets']['MB_infer'] = np.nan
    
    
    for ind in tqdm(snindex):
        t_peak = detectable['targets'].loc[ind].t_peak
        data = detectable['data'][detectable['data'].snindex==ind]
        detectable['targets']['g_peak_mag'].loc[ind], detectable['targets']['r_peak_mag'].loc[ind], detectable['targets']['i_peak_mag'].loc[ind] = obs_peakmag(t_peak, data)
        detectable['targets']['min_peak_mag'].loc[ind] = np.nanmin([detectable['targets'].g_peak_mag.loc[ind], detectable['targets'].r_peak_mag.loc[ind], detectable['targets'].i_peak_mag.loc[ind]])
        
        detectable['targets']['KBr'].loc[ind] = KBRcorr(detectable['targets']['zlens'].loc[ind])
    
        detectable['targets']['MB_infer'].loc[ind] =  detectable['targets'].loc[ind].r_peak_mag - cosmo.distmod(detectable['targets'].loc[ind].zlens).value - detectable['targets'].loc[ind].KBr
    
    
    identifiable = {}
    identifiable['targets'] = detectable['targets'][detectable['targets'].MB_infer<-19.4]
    identifiable['data']    = detectable['data'][np.isin(detectable['data'].snindex, identifiable['targets'].index)]


    return not_observed, not_detected, detectable, identifiable


def format_skysurvey_outputs_unlensed(dsets_list, sntype='ia'):
    not_observed = {'targets': []}

    not_detected = {'data': [],
                    'targets': []}
    
    detectable = {'data': [],
                  'targets': []}
    
    if sntype=='ia':
        add_weight = add_weight_ia
    elif sntype=='iip':
        add_weight = add_weight_iip
    elif sntype=='iin':
        add_weight = add_weight_iin
    elif sntype=='ibc':
        add_weight = add_weight_ibc
    else:
        raise ValueError("sntype undefined")

    
    max_index = 0
    for i in range(len(dsets_list)):
        print(i+1,'/',len(dsets_list))
        dset_ = pd.read_pickle(dsets_list[i])
        data = dset_['data']
        targets = dset_['targets']
        
        #targets['angsep_max'] = [funcmax(targets, i) for i in range(len(targets['z']))]
    
        min_index = max_index + 1
        max_index = min_index + len(targets)
        new_index = np.arange(min_index, max_index, 1) # min_index, max_index inclusive
        old_index = targets.index.values
        targets.set_index(pd.Index(new_index), inplace=True)
    
        data['mags'] = data.zp.values - 2.5 * np.log10(data.flux.values)
        data['magerr'] = 2.5 / np.log(10) * (data.fluxerr.values / data.flux.values)
        data['snr'] = data.flux.values / data.fluxerr.values

        data = data[data['snr']>=3]
        
        data['snindex'] = np.nan

        targets['t_peak'] = time_peak(data)
    
        # We are only saving data from detected sources. ndet>5 epochs around peak
        targets['ztfg_npoints_aroundpeak'] = np.nan
        targets['ztfr_npoints_aroundpeak'] = np.nan
        targets['ztfi_npoints_aroundpeak'] = np.nan
        targets['npoints_aroundpeak'] = np.nan
        
        for ind in tqdm(targets[targets.ndet>=5].index):
            old_index_ = old_index[new_index==ind]
            data['snindex'].loc[old_index_] = ind 
            
            out = get_detections_around_peak(targets.loc[ind].t_peak, 
                                           data.loc[old_index_].time.values, 
                                           data.loc[old_index_].mags.values,
                                           data.loc[old_index_].magerr.values,
                                           data.loc[old_index_].band.values,
                                           data.loc[old_index_].snr.values, snr_threshold=5.)
            npoints_aroundpeak = 0
            for k in out.keys():
                targets[k].loc[ind]  = out[k]
                npoints_aroundpeak += out[k]
            targets['npoints_aroundpeak'].loc[ind] = npoints_aroundpeak
            
        not_observed['targets'].append( targets[np.isnan(targets.ndet)] )
        
        not_detected['targets'].append( targets[~np.isnan(targets.ndet) & (targets.npoints_aroundpeak<5)] )
        
        detectable['targets'].append( targets[targets.npoints_aroundpeak>=5] )
        detectable['data'].append( data[np.isin(data.snindex.values, targets[targets.npoints_aroundpeak>=5].index)] )
    
    not_observed['targets'] = pd.concat(not_observed['targets'])
    not_detected['targets'] = pd.concat(not_detected['targets'])
    detectable['targets'] = pd.concat(detectable['targets'])
    detectable['data'] = pd.concat(detectable['data'], ignore_index=True)


    # Add apparent peak magnitude
    
    snindex = detectable['targets'].index
    
    detectable['targets']['g_peak_mag'] = np.nan
    detectable['targets']['r_peak_mag'] = np.nan
    detectable['targets']['i_peak_mag'] = np.nan
    detectable['targets']['min_peak_mag'] = np.nan
    detectable['targets']['KBr'] = np.nan
    detectable['targets']['MB_infer'] = np.nan
    
    
    for ind in tqdm(snindex):
        t_peak = detectable['targets'].loc[ind].t_peak
        data = detectable['data'][detectable['data'].snindex==ind]
        detectable['targets']['g_peak_mag'].loc[ind], detectable['targets']['r_peak_mag'].loc[ind], detectable['targets']['i_peak_mag'].loc[ind] = obs_peakmag(t_peak, data)
        detectable['targets']['min_peak_mag'].loc[ind] = np.nanmin([detectable['targets'].g_peak_mag.loc[ind], detectable['targets'].r_peak_mag.loc[ind], detectable['targets'].i_peak_mag.loc[ind]])
        
        detectable['targets']['KBr'].loc[ind] = KBRcorr(detectable['targets']['z'].loc[ind])
    
        detectable['targets']['MB_infer'].loc[ind] =  detectable['targets'].loc[ind].r_peak_mag - cosmo.distmod(detectable['targets'].loc[ind].z).value - detectable['targets'].loc[ind].KBr
    
    
    identifiable = {}
    identifiable['targets'] = detectable['targets'][detectable['targets'].MB_infer<-19.5]
    identifiable['data']    = detectable['data'][np.isin(detectable['data'].snindex, identifiable['targets'].index)]


    return not_observed, not_detected, detectable, identifiable

def hostdust_Ia(r_v=2., ebv_rate=0.11, size=None):
    hostr_v = r_v * np.ones(size)
    hostebv = np.random.exponential(ebv_rate, size)
    return hostr_v, hostebv