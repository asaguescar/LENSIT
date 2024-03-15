# Run The LENSIT pipeline to get the detected lightcurves

def parse_commands():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-z', '--zmax', default=1.5, type=float, help='max redshift')
    parser.add_argument('-s', '--size', default=100, type=int, help='size')
    parser.add_argument('-i', '--index', default=0, type=int, help='index')

    return parser.parse_args()

args = parse_commands()
##############################
# Set up the sims:
z_max = args.zmax
size  = args.size
i     = args.index
##############################

from simulations.simulating_lenses import sample_lensing_parameters
from astropy.cosmology import Planck18 as cosmo

out = sample_lensing_parameters(z_max=z_max, cosmo=cosmo, size=size)

import os
path = "../sim_output/"
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created! ", path)


import pickle
outputfilename = path+'lenses_'+str(size)+'_'+str(i)+'.pkl'
with open(outputfilename, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)