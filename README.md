# LENSIT: Lensing End-to-end Supernovae Lightcurve Investigation Tool

This is an end-to-end pipeline to simulate realistic unresolved lensed supernovae lightcurves. 
It combines survey observing logs with a population of synthetic supernova lightcurves, vombines with a distribution of lens properties. 
The assumptions are part of the inputs for the simualtions. Therefore this pipeline allows to investigate effects on the lensing properties to the lightcurves. 

It utilized existing python packages like LENSTRONOMY, to solve the lensing equations and SKYSURVEY, to simulate the lightcurves for the given survey. 

## STEPS:

you simply need to run: 

` python run_skyvision.py --start_data 2018-03-20 --end_date 2022-08-31 `


then we call the survey like.¡:

`import pandas as pd`

`survey = pd.read_pickle('../input/logs/ztf_survey_12.pkl')`

`from skysurvey import ztf`

`ztf_survey = ztf.ZTF(survey, level='quadrant')`

Now to simulate the lenses we run: 

`from src.simulations.simulating_lenses import sample_object_parameters`

`sample_object_parameters(z_max, cosmo, size)`

We should indicate the maximum redshift, the cosmology assumed and the size of the sample that we want to simulate. 

Within the function, it samples the lenses parameters using the functions described for the different parameters and runs Lenstronomy to solve the lensing equation and check if it is strongly lensed.
It takes around 2 minutes to run N=1000. 

We add SN parameters.

We run skysurvey with our sample and the ZTF survey.

# Usage documentation 

Here we explain usage, computational time etc.

- Running function `sample_object_parameters(z_max, cosmo, size)` for z_max=1.5 and  size=10000 takes around 18 minutes. To optimize computing time, one can run this function in parallel according to the CPU power to optimize CPU usage time.

- SNe parameters

- Then we run skysurvey
