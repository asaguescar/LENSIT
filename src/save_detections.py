
def parse_commands():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--lensfile', default='../sim_output/lenses_100_0.pkl', type=str, help='lenses file')

    return parser.parse_args()

args = parse_commands()
##############################
# Set up the sims:
lensfile = args.lensfile
##############################


import pandas as pd
dset = pd.read_pickle(lensfile)
dset['data'] = dset['data'].loc[dset['targets'][dset['targets'].ndet>=2].index]


import pickle
outputfilename = lensfile[:-4]+'_targdata.pkl'
pickle.dump(dset, open(outputfilename, 'wb'))
print('Saved: ', outputfilename)
