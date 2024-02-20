from astropy.cosmology import Planck18 as cosmo
import numpy as np
from scipy.integrate import quad
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Util import param_util
import scipy.special


def psl(z_s, B=1):
    '''PDF of strong lensing per redshift

    Parameters:
    - z_s (float): Redshift value.
    - B (float): Optional parameter (default is 1).

    Returns:
    float: Probability density function value.

    Units:
    - Add information about the units for clarity.
    '''
    z_s_term = 1 + 0.41 * z_s ** 1.1
    result = (5e-4 * z_s ** 3) / z_s_term ** 2.7

    return result


def pzl(z_max, cosmo):
    '''Probability of lensed redshift dependent on cosmological parameters up to a given maximum redshift.

    Parameters:
    - z_max (float): Maximum redshift.
    - cosmo (Cosmology): Cosmological parameters.

    Returns:
    Tuple: Redshift values and corresponding probability density.

    Units:
    - Add information about the units for clarity.
    '''
    z_l_min, z_l_max = 0, z_max
    z_l = np.linspace(z_l_min, z_l_max, 10000)

    def pdf_unnormalized(z_l):
        Dl = cosmo.angular_diameter_distance(z_l).value
        return (1 + z_l) ** 2 * Dl ** 2 / np.sqrt(cosmo.Om0 * (1 + z_l) ** 3 + (1 - cosmo.Om0))

    pdf_unnormalized_ = pdf_unnormalized(z_l)
    norm_cont = quad(pdf_unnormalized, z_l_min, z_l_max)[0]
    pdf_normalized = pdf_unnormalized_ / norm_cont
    pdf_pervolume = np.diff(pdf_normalized)

    # Vectorize the operation for better performance
    z_mean = np.mean([z_l[1:], z_l[:-1]], axis=0)

    return z_mean, pdf_pervolume


def sigma_distr(phi0=8e-3, sigma0=161, alpha=2.32, beta=2.67):
    '''Generate the distribution of velocity dispersion of lens galaxies.

    Parameters:
    - phi0 (float): Parameter (# h^3 Mpc^-3).
    - sigma0 (float): Parameter (km s^-1).
    - alpha (float): Parameter.
    - beta (float): Parameter.

    Returns:
    Tuple: Values of velocity dispersion and corresponding probability density.

    Units:
    - Add information about the units for clarity.
    '''
    x = np.linspace(0.001, 500, 10000)
    pdf = phi0 * (x / sigma0) ** alpha * np.exp(-(x / sigma0) ** beta) * beta / scipy.special.gamma(alpha / beta) / x

    # Normalize the PDF
    pdf_normalized = pdf / np.sum(pdf)

    return x, pdf_normalized


def einstein_radius(z_l, z_s, sigma):
    """
    Calculate the Einstein radius (θ_E) given the velocity dispersion and the distances.

    Parameters:
    - z_l (float): Redshift of the lens.
    - z_s (float): Redshift of the source.
    - sigma (float): Velocity dispersion (in km s^-1).

    Returns:
    - einstein_radius (float): The Einstein radius (in radians).
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


def ellipticity_dispersion_dependent_distr(sigma, A=0.38, B=5.7e-4):
    """
    Generate the ellipticity distribution dependent on the velocity dispersion.

    Parameters:
    - sigma (float): Velocity dispersion (in ---).
    - A (float): Parameter.
    - B (float): Parameter.

    Returns:
    Tuple: Values of ellipticity and corresponding probability density.

    Units:
    - Add information about the units for clarity.
    """
    # Validate sigma
    # if sigma < 0:
    #    raise ValueError("Velocity dispersion (sigma) should be non-negative.")

    x = np.linspace(0, 0.8, 1000)

    s = A + B * sigma
    pdf = x / (s ** 2) * np.exp(-x ** 2 / (s ** 2))

    # Normalize the PDF
    pdf_normalized = pdf / np.sum(pdf)

    return x, pdf_normalized


def shear_distr(shear_max=1.5, s=0.05):
    """
    Generate the shear distribution.

    Parameters:
    - shear_max (float): Maximum shear value.

    Returns:
    Tuple: Values of shear and corresponding probability density.
    """
    x = np.linspace(0, shear_max, 1000)
    pdf = x / (s ** 2) * np.exp(-x ** 2 / (s ** 2))

    # Normalize the PDF
    pdf_normalized = pdf / np.sum(pdf)

    return x, pdf_normalized


def supernova_positions(theta_ein, size=None):
    '''
    Generate random positions for supernovae with respect to the center of the lens galaxy.

    Parameters:
    - theta_ein (float): Einstein radius (in radians).
    - size (int): Number of samples to generate (default is None).

    Returns:
    Tuple: x and y positions of the supernovae.

    Units:
    - Add information about the units for clarity.
    '''
    theta_l = 0.9 * theta_ein
    r = np.random.uniform(0, 1, size=size)
    theta = np.random.uniform(0, 2 * np.pi, size=size)
    x_s = theta_l * np.sqrt(r) * np.cos(theta)
    y_s = theta_l * np.sqrt(r) * np.sin(theta)
    return x_s, y_s


def check_lensing(z_lens, z_source, cosmo, theta_ein, gamma, source_x, source_y, phi_lens, q_lens, phi_gamma):
    # Modelling of the lens galaxies SIE + External SHEAR
    lens_model_list = ['SIE', 'SHEAR']

    # SIE
    e1_lens, e2_lens = param_util.phi_q2_ellipticity(phi_lens, q_lens)
    kwargs_sie = {'theta_E': theta_ein,
                  'e1': e1_lens, 'e2': e2_lens,
                  'center_x': 0., 'center_y': 0.}
    # External shear
    gamma1, gamma2 = param_util.shear_polar2cartesian(phi=phi_gamma,
                                                      gamma=gamma)
    kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}

    lens_model_class = LensModel(lens_model_list, z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    lensEquationSolver = LensEquationSolver(lens_model_class)
    kwargs_lens = [kwargs_sie, kwargs_shear]

    # Checking for multiplicity, meaning that at least 2 images are generated
    x_image, y_image = lensEquationSolver.findBrightImage(float(source_x), float(source_y), kwargs_lens)
    imno = len(x_image)
    multiplicity = imno > 1

    if multiplicity:
        td_images = lens_model_class.arrival_time(x_image=x_image, y_image=y_image, kwargs_lens=kwargs_lens)
        td_images = td_images - min(td_images)
        td_max = td_images.max()
    else:
        td_images = np.nan
        td_max    = np.nan

    # Checking now for total magnification that should be at least 2
    macro_mag = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens)
    mu_i = np.abs(macro_mag)
    mu_total = sum(mu_i)
    magnified = mu_total >= 2

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


def sample_lensing_parameters(z_max=1, cosmo=cosmo, size=None):
    # Sample redshift for strong lensing
    z_s_values = np.random.uniform(0.1, z_max, size=size)
    psl_values = psl(z_s_values)
    # z_s = np.random.choice(z_s_values, size=size, p=psl_values / np.sum(psl_values))
    z_s = z_s_values

    # Sample redshift and probability for lensed redshift
    z_l_values, pzl_values = pzl(z_max, cosmo)

    # Sample parameters for velocity dispersion distribution
    sigma_values_, sigma_pdf = sigma_distr()

    # Sample parameters for shear distribution
    shear_values_, shear_pdf = shear_distr()

    z_l = np.zeros(size)
    sigma_values = np.zeros(size)
    einstein_radius_values = np.zeros(size)
    ellipticity_values = np.zeros(size)
    shear_values = np.zeros(size)
    source_x, source_y = np.zeros(size), np.zeros(size)
    phi_lens, phi_gamma = np.zeros(size), np.zeros(size)
    q_lens = np.zeros(size)
    Lensed = np.zeros(size)
    imno = np.zeros(size)
    mu_total = np.zeros(size)
    td_max = np.zeros(size)
    mu_1, mu_2, mu_3, mu_4 = np.zeros(size), np.zeros(size), np.zeros(size), np.zeros(size)
    dt_1, dt_2, dt_3, dt_4 = np.zeros(size), np.zeros(size), np.zeros(size), np.zeros(size)
    x_image, y_image = [[] for _ in range(size)], [[] for _ in range(size)]

    while sum(Lensed == 1) < size:
        indeces_nolensing = np.where(Lensed == 0)[0]
        for ind in indeces_nolensing:
            while True:
                # Ensure z_l is always smaller than z_s
                # Sample z_l until it satisfies the condition
                z_l[ind] = np.random.choice(z_l_values, p=pzl_values / np.sum(pzl_values))
                if z_l[ind] < z_s[ind]:
                    break
        sigma_values[indeces_nolensing] = np.random.choice(sigma_values_, size=len(indeces_nolensing),
                                                           p=sigma_pdf / np.sum(sigma_pdf))

        # Sample parameters for Einstein radius calculation
        einstein_radius_values[indeces_nolensing] = einstein_radius(z_l[indeces_nolensing], z_s[indeces_nolensing],
                                                                    sigma_values[indeces_nolensing])

        # Sample parameters for ellipticity distribution

        for ind in indeces_nolensing:
            ellipticity_values_, ellipticity_pdf = ellipticity_dispersion_dependent_distr(sigma_values[ind])
            ellipticity_values[ind] = float(np.random.choice(ellipticity_values_, size=1, p=ellipticity_pdf))

        shear_values[indeces_nolensing] = np.random.choice(shear_values_, size=len(indeces_nolensing), p=shear_pdf)

        # Sample supernova positions
        source_x[indeces_nolensing], source_y[indeces_nolensing] = supernova_positions(
            einstein_radius_values[indeces_nolensing], size=len(indeces_nolensing))

        phi_lens[indeces_nolensing] = np.random.uniform(0, 2 * np.pi, size=len(indeces_nolensing))
        q_lens[indeces_nolensing] = 1 - ellipticity_values[indeces_nolensing]
        phi_gamma[indeces_nolensing] = np.random.uniform(0, 2 * np.pi, size=len(indeces_nolensing))

        for ind in indeces_nolensing:
            Lensed[ind], imno[ind], x_image[ind], y_image[ind], mu_total[ind], td_max[ind], mu_1[ind], mu_2[ind], mu_3[
                ind], mu_4[ind], dt_1[ind], \
            dt_2[ind], dt_3[ind], dt_4[ind] = check_lensing(z_l[ind], z_s[ind], cosmo, einstein_radius_values[ind],
                                                            shear_values[ind],
                                                            source_x[ind],
                                                            source_y[ind], phi_lens[ind], q_lens[ind], phi_gamma[ind])

    return {
        'z': z_s,
        'psl_values': psl_values,
        'z_l': z_l,
        'sigma_values': sigma_values,
        'einstein_radius_values': einstein_radius_values,
        'ellipticity_values': ellipticity_values,
        'gamma': shear_values,
        'source_x': source_x,
        'source_y': source_y,
        'x_image': x_image,
        'y_image': y_image,
        'phi_lens': phi_lens,
        'q_lens': q_lens,
        'phi_gamma': phi_gamma,
        'imno': imno,
        'mu_total': mu_total,
        'td_max': td_max,
        'mu_1': mu_1, 'mu_2': mu_2, 'mu_3': mu_3, 'mu_4': mu_4,
        'dt_1': dt_1, 'dt_2': dt_2, 'dt_3': dt_3, 'dt_4': dt_4,
        'Lensed': Lensed
    }