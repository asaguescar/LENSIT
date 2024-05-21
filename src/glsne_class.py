import sncosmo
import numpy as np

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


# Example usage
if __name__ == "__main__":
    data = ...  # Load your data
    model = GLSNe()
