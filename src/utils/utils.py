# I want to combine the params and the targets
import glob
import pandas as pd

def combine_target_and_lcparams(globinput = '../sim_output/dset_ia_salt2_10000_?.pkl'):
    dsetfiles = glob.glob(globinput)
    dsets = []
    for dset_ in dsetfiles:
        df_ = pd.read_pickle(dset_)
        lcpar_ = pd.read_pickle(dset_[:-4]+'_lcparams.pkl')
        dfcomb_ = pd.concat([df_['targets'], lcpar_], axis=1)
        dfcomb_ = dfcomb_.loc[:, ~dfcomb_.columns.duplicated()] # to eliminate duplicated columns
        dsets.append(dfcomb_)
    return pd.concat(dsets, ignore_index=True)
