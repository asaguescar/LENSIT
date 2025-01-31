from astropy.cosmology import Planck18 as cosmo
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Util import param_util
import scipy.special
from tqdm import tqdm
import pandas as pd
from scipy.integrate import dblquad, quad


class LensGalaxy:
    """
    A class for simulating lens galaxy properties and calculating the probability of lensing events.
    """

    def pzl_pdf(self, z, cosmo=cosmo):
        """
        Calculates the probability of being a lens for given redshifts based on differential comoving volume.

        Parameters:
        - z (array_like): Array of redshift values.
        - cosmo (Cosmology): An astropy cosmology instance.

        Returns:
        - array_like: Probability density function values.
        """
        Dl = cosmo.angular_diameter_distance(z).value # Angular diameter at the lens redshift
        pdf = (1 + np.array(z)) ** 2 * Dl ** 2 / cosmo.efunc(z)
        return pdf


    def sigma_pdf(self, sv, phi0=2.099, sigma0=113.78, alpha=0.94, beta=1.85):
        """
        Generates the probability density function for the velocity dispersion of lens galaxies using
        the model parameters from Bernardi 2010.

        Parameters:
        - sv (array_like): Array of velocity dispersions.
        - phi0 (float): Normalization constant (# h^3 Mpc^-3).
        - sigma0 (float): Characteristic velocity dispersion (km s^-1).
        - alpha (float): Power-law slope.
        - beta (float): Exponential cutoff steepness.

        Returns:
        - array_like: Probability density function values.
        """
        sv = np.array(sv)
        pdf = phi0 * (sv / sigma0) ** alpha * np.exp(-(sv / sigma0) ** beta) * beta / scipy.special.gamma(
            alpha / beta)
        return pdf

    def shear_pdf(self, gamma, s=0.05):
        """
        Generates the probability density function of shear values.

        Parameters:
        - gamma (array_like): Array of shear values.
        - s (float): Dispersion of the shear values.

        Returns:
        - array_like: Probability density function values.
        """
        gamma = np.array(gamma)
        pdf = gamma / (s ** 2) * np.exp(-gamma ** 2 / (2*s ** 2))

        return pdf

    def ellipticity_pdf(self, e, sigma, A=0.38, B=5.7e-4):
        """
        Generates the probability density function for ellipticity based on lens velocity dispersion.

        Parameters:
        - e (array_like): Array of ellipticity values.
        - sigma (float): Velocity dispersion used to calculate scale parameter.
        - A (float): Scale parameter for the base ellipticity distribution.
        - B (float): Scale factor for dependency on velocity dispersion.

        Returns:
        - array_like: Probability density function values.

        Reference: Collett 2015
        """
        e = np.array(e)
        s = A + B * sigma
        pdf = e / (s ** 2) * np.exp(-e ** 2 / (2 * s ** 2))
        return pdf

    def supernova_positions(self, theta_ein, size=None):
        """
        Generates random positions for supernovae around the lens galaxy center.

        Parameters:
        - theta_ein (float): Einstein radius in radians.
        - size (int, optional): Number of position samples to generate (default is None).

        Returns:
        - tuple: Arrays of x and y positions.
        """
        theta_l = theta_ein
        r = np.random.uniform(0, 1, size=size)
        theta = np.random.uniform(0, 2 * np.pi, size=size)
        x_s = theta_l * np.sqrt(r) * np.cos(theta)
        y_s = theta_l * np.sqrt(r) * np.sin(theta)
        return x_s, y_s

    def einstein_radius(self, z_l, z_s, sigma):
        """
        Calculates the Einstein radius given the lens and source redshifts and the lens velocity dispersion.

        Parameters:
        - z_l (float): Redshift of the lens.
        - z_s (float): Redshift of the source.
        - sigma (float): Velocity dispersion of the lens (km s^-1).

        Returns:
        - float: Einstein radius in arcseconds.
        """
        # Convert velocity dispersion to m/s
        sigma_m = sigma * 1e3

        # Speed of light in a vacuum (m/s)
        speed_of_light = 3e8

        # Convert distances to meters
        distance_lens_source = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to('m').value
        distance_source = cosmo.angular_diameter_distance(z_s).to('m').value

        # Calculate the Einstein radius
        einstein_radius_radian = 4 * np.pi * (sigma_m / speed_of_light) ** 2 * distance_lens_source / distance_source

        # Radians to arcseconds
        einstein_radius = einstein_radius_radian * 180 * 3600 / np.pi

        return einstein_radius

    def sample(self, z_max=2, size=None):
        """
        Samples lens properties such as redshift, velocity dispersion, shear, and ellipticity.

        Parameters:
        - z_max (float): Maximum redshift for sampling.
        - size (int, optional): Number of samples (default is None).

        Returns:
        - dict: Dictionary containing sampled lens properties.
        """

        x = np.linspace(0, z_max, 1000)
        pdf = self.pzl_pdf(x)
        zlens = np.random.choice(x, size=size, p=pdf/np.sum(pdf))

        x = np.linspace(50,400, 1000)
        pdf = self.sigma_pdf(x)
        sigma = np.random.choice(x, size=size, p=pdf / np.sum(pdf))

        x = np.linspace(0,1.5, 1000)
        pdf = self.shear_pdf(x)
        gamma = np.random.choice(x, size=size, p=pdf / np.sum(pdf))

        e = []
        for i in range(size):
            x = np.linspace(0,0.8, 1000)
            pdf = self.ellipticity_pdf(x, sigma[i])
            e.append( float( np.random.choice(x, size=1, p=pdf / np.sum(pdf))) )

        return {'zlens':zlens,
                'sigma':sigma,
                'gamma':gamma,
                'ellipticity':e,
                'phi_lens': np.random.uniform(0,2*np.pi, size),
                'phi_gamma': np.random.uniform(0,2*np.pi, size)
                }

    def lenstronomy(self, zlens, zsource, phi_lens, ellipticity, phi_gamma, gamma, theta_ein, source_x, source_y, cosmo=cosmo, mu_total_min=2):
        """
        Simulates the lensing effect including multiple images and checks for total magnification.

        Parameters:
        - zlens, zsource: Redshifts of the lens and the source.
        - phi_lens, phi_gamma: Angular positions of the lens and external shear.
        - ellipticity, gamma: Ellipticity of the lens and shear.
        - theta_ein: Einstein radius.
        - source_x, source_y: Positions of the source.
        - cosmo: Cosmology used for the simulation.
        - mu_total_min (float): Minimum total magnification.

        Returns:
        - tuple: Contains lensing boolean, number of images, image positions, total magnification, and time delays.
        """
        # Modelling of the lens galaxies SIE + External SHEAR
        lens_model_list = ['SIE', 'SHEAR']

        # SIE
        e1_lens, e2_lens = param_util.phi_q2_ellipticity(phi_lens, 1-ellipticity)
        kwargs_sie = {'theta_E': theta_ein,
                      'e1': e1_lens, 'e2': e2_lens,
                      'center_x': 0., 'center_y': 0.}
        # External shear
        gamma1, gamma2 = param_util.shear_polar2cartesian(phi=phi_gamma,
                                                          gamma=gamma)
        kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}

        lens_model_class = LensModel(lens_model_list, z_lens=zlens, z_source=zsource, cosmo=cosmo)
        lensEquationSolver = LensEquationSolver(lens_model_class)
        kwargs_lens = [kwargs_sie, kwargs_shear]

        # Checking for multiplicity, meaning that at least 2 images are generated
        x_image, y_image = lensEquationSolver.findBrightImage(source_x, source_y, kwargs_lens)
        imno = len(x_image)
        multiplicity = imno > 1

        if multiplicity:
            td_images = lens_model_class.arrival_time(x_image=x_image, y_image=y_image, kwargs_lens=kwargs_lens)
            td_images = td_images - min(td_images)
            td_max = td_images.max()
        else:
            td_images = np.nan
            td_max = np.nan

        # Checking now for total magnification that should be at least 2
        macro_mag = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens)
        mu_i = np.abs(macro_mag)
        mu_total = sum(mu_i)
        magnified = mu_total >= mu_total_min

        lensed = magnified and multiplicity
        try:
            mu_1 = mu_i[0]
            dt_1 = td_images[0]
        except:
            mu_1, dt_1 = 0, 0
        try:
            mu_2 = mu_i[1]
            dt_2 = td_images[1]
        except:
            mu_2, dt_2 = 0, 0
        try:
            mu_3 = mu_i[2]
            dt_3 = td_images[2]
        except:
            mu_3 = 0
            dt_3 = 0
        try:
            mu_4 = mu_i[3]
            dt_4 = td_images[3]
        except:
            mu_4 = 0
            dt_4 = 0

        return lensed, imno, x_image, y_image, mu_total, td_max, mu_1, mu_2, mu_3, mu_4, dt_1, dt_2, dt_3, dt_4



    def sample_lensed_zs(self, zsource=2, size=None, mu_total_min=2):
        """
        Samples lensed positions based on source redshift and minimum total magnification.

        Parameters:
        - zsource (float): Redshift of the source.
        - size (int, optional): Number of samples.
        - mu_total_min (float): Minimum total magnification.

        Returns:
        - dict: Lensed positions and properties.
        """
        lenses = self.sample(z_max=zsource, size=size)
        lenses['theta_ein'] = self.einstein_radius(lenses['zlens'], zsource, lenses['sigma'])
        lenses['source_x'], lenses['source_y'] = self.supernova_positions(lenses['theta_ein'], size=size)
        lenses['zsource'] = np.ones(size)*zsource

        lenses['imno'] = []
        lenses['x_image'] = []
        lenses['y_image'] = []
        lenses['mu_total'] = []
        lenses['td_max'] = []
        lenses['mu_1'] = []
        lenses['mu_2'] = []
        lenses['mu_3'] = []
        lenses['mu_4'] = []
        lenses['dt_1'] = []
        lenses['dt_2'] = []
        lenses['dt_3'] = []
        lenses['dt_4'] = []
        for i in range(size):
            q = self.lenstronomy(lenses['zlens'][i],
                            lenses['zsource'][i],
                            lenses['phi_lens'][i],
                            lenses['ellipticity'][i],
                            lenses['phi_gamma'][i],
                            lenses['gamma'][i],
                            lenses['theta_ein'][i],
                            lenses['source_x'][i],
                            lenses['source_y'][i], mu_total_min=mu_total_min)
            if q[0]:
                lenses['imno'].append(q[1])
                lenses['x_image'].append(q[2])
                lenses['y_image'].append(q[3])
                lenses['mu_total'].append(q[4])
                lenses['td_max'].append(q[5])
                lenses['mu_1'].append(q[6])
                lenses['mu_2'].append(q[7])
                lenses['mu_3'].append(q[8])
                lenses['mu_4'].append(q[9])
                lenses['dt_1'].append(q[10])
                lenses['dt_2'].append(q[11])
                lenses['dt_3'].append(q[12])
                lenses['dt_4'].append(q[13])
            else:
                while q[0]==False:
                    lens_ = self.sample(z_max=zsource, size=1)
                    lens_['theta_ein'] = self.einstein_radius(lens_['zlens'], zsource, lens_['sigma'])
                    lens_['source_x'], lens_['source_y'] = self.supernova_positions(lens_['theta_ein'], size=1)
                    lens_['zsource'] = zsource

                    q = self.lenstronomy(float(lens_['zlens']),
                                         float(lens_['zsource']),
                                         float(lens_['phi_lens']),
                                         float(lens_['ellipticity'][0]),
                                         float(lens_['phi_gamma']),
                                         float(lens_['gamma']),
                                         float(lens_['theta_ein']),
                                         float(lens_['source_x']),
                                         float(lens_['source_y']))
                    if q[0]:
                        lenses['imno'].append(q[1])
                        lenses['x_image'].append(q[2])
                        lenses['y_image'].append(q[3])
                        lenses['mu_total'].append(q[4])
                        lenses['td_max'].append(q[5])
                        lenses['mu_1'].append(q[6])
                        lenses['mu_2'].append(q[7])
                        lenses['mu_3'].append(q[8])
                        lenses['mu_4'].append(q[9])
                        lenses['dt_1'].append(q[10])
                        lenses['dt_2'].append(q[11])
                        lenses['dt_3'].append(q[12])
                        lenses['dt_4'].append(q[13])

        return lenses

    
    def ASL(self, z_l, z_s, sigma, lensmodel='SIE'):
        """
        Calculates the lensing cross-section area for a given lens model.

        Parameters:
        - z_l (float): Redshift of the lens.
        - z_s (float): Redshift of the source.
        - sigma (float): Velocity dispersion of the lens (km s^-1).
        - lensmodel (str): Lens model type, default is 'SIE' (Singular Isothermal Ellipsoid).

        Returns:
        - float: Lensing cross-section in square arcseconds.
        """
        if lensmodel == 'SIE':
            theta_ein = self.einstein_radius(z_l, z_s, sigma) 
            theta_ein_rad = theta_ein * np.pi / (180 * 3600)
            return np.pi * theta_ein_rad ** 2

    
    def Psl(self, zs, B=1):
        """
        Calculates the lensing probability for a range of source redshifts.

        Parameters:
        - zs (float): Source redshift.
        - B (float): Bias factor, default is 1.

        Returns:
        - float: Integrated lensing probability over the source redshift.
        """
        int_sigma = quad(self.sigma_pdf, 0, np.inf)[0]

        def ASL(z_l, z_s, sigma, lensmodel='SIE'):
            '''lensing cross-section'''
            if lensmodel == 'SIE':
                theta_ein = self.einstein_radius(z_l, z_s, sigma)
                theta_ein_rad = theta_ein * np.pi / (180 * 3600)
                return np.pi * theta_ein_rad ** 2

        def integral_db(z_l, sigma):
            return cosmo.differential_comoving_volume(z_l).to('Gpc3/sr').value * self.sigma_pdf(
                sigma) / int_sigma * B * ASL(z_l, zs, sigma) / np.pi * (180 * 3600)

        return dblquad(integral_db, 0, np.inf, 0, zs)[0]

    def dif_Psl(self, zs, z_l, sigma, B=1):
        """
        Calculates differential lensing probability for a given lens redshift and velocity dispersion.

        Parameters:
        - zs (float): Source redshift array.
        - z_l (float): Lens redshift.
        - sigma (float): Velocity dispersion of the lens (km s^-1).
        - B (float): Bias factor, default is 1.

        Returns:
        - float: Differential lensing probability.
        """
        int_sigma = quad(lens.sigma_pdf, 0, np.inf)[0]

        def ASL(z_l, z_s, sigma, lensmodel='SIE'):
            '''lensing cross-section'''
            if lensmodel == 'SIE':
                theta_ein = [lens.einstein_radius(z_l, z, sigma) for z in z_s]
                theta_ein_rad = theta_ein * np.pi / (180 * 3600)
                return np.pi * theta_ein_rad ** 2

        def integral_db(z_l, sigma):
            return cosmo.differential_comoving_volume(z_l).to('Gpc3/sr').value * lens.sigma_pdf(
                sigma) / int_sigma * B * ASL(z_l, zs, sigma) / np.pi * (180 * 3600)

        return integral_db(z_l, sigma)

    def psl_simpl(self, z_s, B=1):
        """
        Provides a simplified probability density function of strong lensing per source redshift.

        Parameters:
        - z_s (float): Source redshift.
        - B (float): Bias factor, default is 1.

        Returns:
        - float: Simplified lensing probability density.
        """
        z_s_term = 1 + 0.41 * z_s ** 1.1
        result = (5e-4 * z_s ** 3) / z_s_term ** 2.7

        return result

    def sample_uniform_zs(self, z_min=0.1, z_max=2.0, size = 1000, mu_total_min=2):
        """
        Samples uniform redshifts and generates lensing events based on those redshifts.

        Parameters:
        - z_min (float): Minimum redshift value.
        - z_max (float): Maximum redshift value.
        - size (int): Number of redshifts to sample.
        - mu_total_min (float): Minimum total magnification for lensing to be considered significant.

        Returns:
        - DataFrame: Dataframe containing the properties of lensed events.
        """
        zs = np.random.uniform(z_min, z_max, size=size)

        dflenses = []
        for i in tqdm(range(len(zs))):
            # print(zs[i])
            lenses = self.sample_lensed_zs(zsource=zs[i], size=1, 
                                           mu_total_min=mu_total_min)  # Returns strongly lensed events: at lleast 2 images and minimum total magnification 2.
            df = pd.DataFrame(lenses)
            dflenses.append(df)
        return  pd.concat(dflenses, ignore_index=True)





    