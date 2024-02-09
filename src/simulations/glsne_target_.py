import sncosmo
import numpy as np
from skysurvey import target
#from ztf_sims import rate_glIa, zlens_from_pdf, random_mutotal, inmo_dt_dist, redshift_Ia_DG, dts_dist, mus_dist, x1_pdf, c_pdf, magabs_Ia, magabs_magnified, fromsfdmaps, hostdust_Ia, rate_glCC, magabs_IIP, rate_Ia, sample_from_DG, zs_from_DG
import pandas as pd
from astropy.cosmology import Planck18
from skysurvey.tools import utils

# Here we are going to define the  models an parameter distributions for skysurvey. 

# First we are going to define the sncosmo model source class to simulate strongly lensed supernova lightcurves. 

dust = sncosmo.CCM89Dust()

class GLSNe(sncosmo.Source):
    '''
    GLSNe is a class for gravitational lensed supernovae models.

    Args:
        sntype (str): Type of supernova model (default: "salt2").
        nimages (int): Number of multiple images (default: 2).
        name (str): Name of the model (optional).
        versifon (str): Version of the model (optional).
    '''

    def __init__(self, sntype="salt2", nimages=2, name=None, version=None):
        self.name = name
        self.version = version
        self._sntype = sntype
        self._source = sncosmo.get_source(sntype)
        self._nimages = nimages

        # Define lensed parameters
        self._parameters = list(np.concatenate([[0, 1] for k in range(1, nimages + 1)]))
        self._param_names = list(np.concatenate([['dt_%i' % k, 'mu_%i' % k] for k in range(1, nimages + 1)]))
        self.param_names_latex = list(np.concatenate([['dt_%i' % k, 'mu_%i' % k] for k in range(1, nimages + 1)]))

        # Add SN parameters from the source
        self._parameters.extend(self._source._parameters)
        self._param_names.extend(self._source._param_names)
        self.param_names_latex.extend(self._source.param_names_latex)

        self._current_parameters = self._parameters.copy()
        
        # Me trying things. Works? This should add the shorter wavelength and assign them 0 flux
        extra_wave = np.arange(1000, 1700, 10)
        self._source._wave = np.concatenate([extra_wave,self._source._wave])
        
    def minwave(self):
        return self._source.minwave()

    def maxwave(self):
        return self._source.maxwave()

    def minphase(self):
        return self._source.minphase()

    def maxphase(self):
        return self._source.maxphase()

    def update_param(self):
        param_tmp = list(self._parameters[self._nimages * 2:])
        for n_ in self._source._param_names:
            self._source.set(**{n_: param_tmp.pop(0)})

        self._current_parameters = self._parameters.copy()

    def _flux(self, phase, wave):
        if np.any(self._current_parameters != self._parameters):
            self.update_param()

        out = np.zeros((len(phase), len(wave)))

        for k in range(0, self._nimages * 2, 2):
            if k==0:
                dt = self._parameters[k]
                mu = self._parameters[k + 1]
                out[:] = self._source._flux(phase - dt, wave) * mu
            else:
                dt = self._parameters[k]
                mu = self._parameters[k + 1]
                out[:] += self._source._flux(phase - dt, wave) * mu

        return out


# For using with skysurvey we need to create a class to define the model and the parameters distribution. 

from src.simulations.simulating_lenses import psl

def rate_glIa(z, exp=1.5):
    '''rate per redshift for Ia supernovae:
    return: rate in Gpc-3 yr-1
    '''
    return 2.35e4 * (1.+z)**exp * psl(z)


class GLSNeIa_salt2( target.core.Transient ):
    _KIND = "GLSNIa"
    sntemplate = "salt2-extended"
    source = GLSNe(sntemplate, 4)
    _TEMPLATE = sncosmo.Model(source, effects=[dust, dust, dust], effect_names=['MW', 'host', 'lens'], effect_frames=['obs', 'rest', 'free'])

    _RATE = staticmethod(rate_glIa)

    _MODEL = dict(redshift={"param": {"zmax": 2}, "as": "z"},
                  zlens={"func": 0, "kwargs":{"z": "@z"}, "as":"zlens"},
                  mu_total={"func": 0, "kwargs":{"z":"@z"}},
                  inmodt={"func": 0, "kwargs":{"z":"@z", "zlens":"@zlens", "mu_total":"@mu_total"},
                          "as":["imno", "dt_max"]}, # We use Danny's catlog to distribute the time delays and multiplicity.
                  dts={"func": 0,  "kwargs":{"imno":"@imno", "dt_max":"@dt_max"},
                       "as": ["dt_1", "dt_2", "dt_3", "dt_4"]}, # Now we want to get the individual time delays.
                  mus={"func": 0,  "kwargs":{"imno":"@imno", "mu_total":"@mu_total"},
                       "as": ["mu_1", "mu_2", "mu_3", "mu_4"]}, # Now we want to get the individual time delays.
                  x1={"func": 0},
                  c={"func": 0},
                  t0={"func": np.random.uniform,
                      "kwargs": {"low": 0, "high": 1000}},
                  magabs_intrinsic = {"func": 0,
                             "kwargs": {"x1": "@x1", "c": "@c",
                                        "mabs":-19.3, "sigmaint":0.10}},
                  magabs_lensed = {"func": 0,
                             "kwargs": {"magabs": "@magabs_intrinsic", "mu_total": "@mu_total"}},
                  magobs={"func": "magabs_to_magobs",
                          "kwargs": {"z":"@z", "magabs":"@magabs_lensed"}},
                  x0={"func": "mag_to_x0",
                      "kwargs": {"z":"@z", "magabs_intrinsic":"@magabs_intrinsic", "sntemplate":0}},
                  radec={"func": utils.random_radec,
                            "kwargs": dict(ra_range=[0, 360], dec_range=[-30, 90]),
                     "as":["ra","dec"]}, # random_radec returns a list, here are the names,
                  mwebv={"func": 0,
                         "kwargs": {"ra":"@ra", "dec":"@dec"}},
                  hostebvr_v={"func": 0,
                              "as" :["hostr_v", "hostebv"]}
                  #lensebvr_v={"func": lensdust_Ia,
                  #            "as" :["lensr_v", "lensebv"]}
                 )

    def mag_to_x0(self, z, magabs_intrinsic, sntemplate="salt2-extended", band="bessellb",zpsys="ab", cosmology=Planck18):
        """ """
        mapp_unlensed = cosmology.distmod(np.asarray(z, dtype="float32")).value + magabs_intrinsic # the apparent magnitude if unlensed
        template = sncosmo.Model(sntemplate)
        m_current = template.source_peakmag("bessellb","vega")
        #m_current = -19.3
        return 10.**(0.4 * (m_current - mapp_unlensed)) * template.get("x0")


