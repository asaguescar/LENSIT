import sncosmo
import numpy as np

class GLSNe(sncosmo.Source):
    """
    A class for simulating gravitational lensed supernovae models using SNCosmo.
    
    Attributes:
        sntype (str): Type of the supernova model, default is 'salt2'.
        nimages (int): Number of images due to lensing, default is 2.
        name (str, optional): Name of the model.
        version (str, optional): Version of the model.
    """

    def __init__(self, sntype="salt2", nimages=2, name=None, version=None):
        """
        Initializes the GLSNe model with specified supernova type and lensing images.

        Parameters:
            sntype (str): Type of supernova model (default: "salt2").
            nimages (int): Number of lensed images (default: 2).
            name (str, optional): Name of the model.
            version (str, optional): Version of the model.
        """
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
        """Returns the minimum wavelength of the model."""
        return self._source.minwave()

    def maxwave(self):
        """Returns the maximum wavelength of the model."""
        return self._source.maxwave()

    def minphase(self):
        """Returns the minimum phase of the model."""
        return self._source.minphase()

    def maxphase(self):
        """Returns the maximum phase of the model."""
        return self._source.maxphase()

    def update_param(self):
        """
        Updates the model parameters from the underlying supernova model if they have changed.
        """
        param_tmp = list(self._parameters[self._nimages * 2:])
        for n_ in self._source._param_names:
            self._source.set(**{n_: param_tmp.pop(0)})

        self._current_parameters = self._parameters.copy()

    def _flux(self, phase, wave):
        """
        Calculates the flux for the given phase and wavelength, accounting for lensing effects.

        Parameters:
            phase (array_like): Phases at which to calculate the flux.
            wave (array_like): Wavelengths at which to calculate the flux.

        Returns:
            array_like: Flux values adjusted for lensing magnifications and delays.
        """
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
