import os.path

from ztfquery import skyvision
import pandas as pd
import numpy as np
from astropy.time import Time
import time

# 1# We first get a list of dates in the adequate format in the range of interest.
from datetime import datetime, timedelta


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def generate_date_range(start_date, end_date):
    date_list = []
    current_date = start_date

    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return date_list

def parse_commands():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start_date', default='2018-03-20', type=str, help='starting date for the query')
    parser.add_argument('-e', '--end_date'  , default='2022-08-31', type=str, help='ending date for the query')

    return parser.parse_args()

args = parse_commands()

# Define your date range

start_date = args.start_date.split('-')
end_date   = args.end_date.split('-')

start_date = datetime(int(start_date[0]), int(start_date[1]), int(start_date[2]))
end_date   = datetime(int(end_date[0]), int(end_date[1]), int(end_date[2]))

print('start_date: ', start_date)
print('end_date: ', end_date)

date_range = generate_date_range(start_date, end_date)

summary_values = ['obsdatetime', 'obsjd', 'expid', 'field', 'rcid', 'fid',
                  'scigain', 'pabszp', 'maglimcat', 'diffmaglim', 'maglimit', 'infobits', 'status',
                  'exptime', 'sciinpseeing', 'difffwhm', 'programid', 'fwhm', 'statusdif', 'qcomment']

loglist = []
datalen = 0
ifile = 0

l = len(date_range)
printProgressBar(0, l, prefix='Progress:', suffix='Complete', length=50)

logspath = '../input/logs/'
if os.path.isdir(logspath) is False:
    os.makedirs(logspath)

for i, nightTime in enumerate(date_range):
    table = skyvision.download_qa_log(nightTime, summary_values=summary_values, store=False)
    loglist.append(table)
    datalen += len(table)
    # -- If long enough write data
    if datalen >= 4_000_000:
        print(nightTime)
        df = pd.concat(loglist)
        df.to_parquet(logspath+f'skylog_{ifile}.parquet')
        loglist = []
        datalen = 0
        ifile += 1
    time.sleep(0.1)
    # Update Progress Bar
    printProgressBar(i + 1, l, prefix='Progress:', suffix='Complete', length=50)

df = pd.concat(loglist)
df.to_parquet(logspath+f'skylog_{ifile}.parquet', index=False)

# Clean log file (based on Bastien's code)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ztfquery import skyvision
import glob
from astropy.time import Time

logfiles = glob.glob(logspath+'skylog_*')
print(logfiles)
dtype = {'field': np.uint16, 'rcid': np.uint8, 'fid': np.uint8, 'expid': np.uint32,
         'obsjd': np.float64, 'scigain': np.float32,
         'maglimcat': np.float32, 'diffmaglim': np.float32, 'maglimit': np.float32,
         'pabszp': np.float32, 'status': 'bool', 'qcomment': "S16"}
Skyvislogs = pd.concat([pd.read_parquet(f) for f in logfiles])
Skyvislogs = Skyvislogs.astype(dtype=dtype)
Skyvislogs.sort_values('obsjd', inplace=True)
Skyvislogs.reset_index(inplace=True)
Skyvislogs.drop(columns='index', inplace=True)

## Remove NaN infobits

NaNinfbitsmask = ~Skyvislogs.infobits.isna()
print(f'Pass cut : {np.sum(NaNinfbitsmask) / len(Skyvislogs)* 100:.2f} %')

## Status mask

statusmask = Skyvislogs['status'] == 1
print(f'Pass cut : {np.sum(statusmask) / len(Skyvislogs)* 100:.2f} %')

## Gain Mask

gainmask = Skyvislogs['obsjd'] > 2458288 # From this date it seems to be clean
print(f'Pass cut : {np.sum(gainmask) / len(Skyvislogs)* 100:.2f} %')

## Apply first masks

datadf = Skyvislogs[gainmask & statusmask & NaNinfbitsmask].copy()
datadf['infobits'] = datadf['infobits'].astype(np.uint64)

rcid_gaindic = {}
for rcid in datadf.rcid.unique():
    rcidf = datadf['scigain'][(datadf.rcid == rcid) & ~datadf.scigain.isna()]
    change_in_gain = rcidf.ne(rcidf.shift()).index[rcidf.ne(rcidf.shift())].to_list()
    rcid_gaindic[rcid] = [datadf.loc[i].obsjd for i in change_in_gain]

first_gain_df = datadf[datadf.obsjd < 2458778.5]  # Seems to have a gain change at this date
gain_map = {}
nonnan_first_gain_df = first_gain_df[~first_gain_df.scigain.isna()]
for rcid in first_gain_df.rcid.unique():
    gain_map[rcid] = nonnan_first_gain_df['scigain'][(nonnan_first_gain_df.rcid == rcid)].iloc[0]

mask = (datadf.obsjd < 2458778.5) & datadf.scigain.isna()
datadf['scigain'][mask] = datadf[mask].rcid.map(gain_map)

second_gain_df = datadf[datadf.obsjd >= 2458778.5]  # Seems to have a gain change at this date
gain_map = {}
nonnan_second_gain_df = second_gain_df[~second_gain_df.scigain.isna()]
for rcid in nonnan_second_gain_df.rcid.unique():
    gain_map[rcid] = nonnan_second_gain_df['scigain'][nonnan_second_gain_df.rcid == rcid].iloc[0]

mask = (datadf.obsjd >= 2458778.5) & datadf.scigain.isna()
datadf['scigain'][mask] = datadf[mask].rcid.map(gain_map)



# Format for the skysurvey

### Convert jd to mjd

datadf['expMJD'] = datadf.obsjd - 2400000.5
datadf['expMJD'] = datadf.expMJD.astype(np.float32)

### Convert filter name

fid_dic = {1: 'ztfg', 2:'ztfr', 3:'ztfi'} # From ztf pipeline deliverable
datadf['filter'] = datadf['fid'].map(fid_dic)

### Field Ra, Dec

from ztfquery import fields
field_RA_dic = {}
field_Dec_dic = {}
no_fields = []
for f in datadf['field'].unique():
    try:
        field_RA_dic[f] = np.radians(fields.get_field_centroid(f)[0][0])
        field_Dec_dic[f] = np.radians(fields.get_field_centroid(f)[0][1])
    except:
        if f not in no_fields:
            no_fields.append(f)

print(f"Fields without coord : {datadf['field'].isin(no_fields).sum() / len(datadf['field']) * 100:.5f} %")

datadf = datadf[~datadf.field.isin(no_fields)].copy()
datadf.rename(columns={'field': 'fieldID', 'pabszp': 'zp', 'scigain': 'gain'}, inplace=True)

datadf['fieldRA'] = datadf.fieldID.map(field_RA_dic).astype(np.float32)
datadf['fieldDec'] = datadf.fieldID.map(field_Dec_dic).astype(np.float32)

datadf.to_parquet(logspath+'ztf_obsfile.parquet')

Obsdf_maglimcat = datadf[['expMJD', 'filter', 'fieldID', 'fieldRA', 'fieldDec', 'rcid', 'maglimcat', 'zp', 'gain', 'expid', 'infobits',
                         'exptime', 'fwhm', 'qcomment', 'programid']]
Obsdf_maglimcat.reset_index(inplace=True, drop=True)
Obsdf_maglimcat.expMJD.is_monotonic_increasing
Obsdf_maglimcat.to_parquet(logspath+'ztf_obsfile_maglimcat_new.parquet')

Obsdf_diffmaglim = datadf[['expMJD', 'filter', 'fieldID', 'fieldRA', 'fieldDec', 'rcid', 'diffmaglim', 'zp', 'gain', 'expid', 'infobits',
                         'exptime', 'fwhm', 'qcomment', 'programid']]
Obsdf_diffmaglim.reset_index(inplace=True, drop=True)
Obsdf_diffmaglim.expMJD.is_monotonic_increasing
Obsdf_diffmaglim.to_parquet(logspath+'ztf_obsfile_diffmaglim_new.parquet')

Obsdf_maglimit = datadf[['expMJD', 'filter', 'fieldID', 'fieldRA', 'fieldDec', 'rcid', 'maglimit', 'zp', 'gain', 'expid', 'infobits',
                         'exptime', 'fwhm', 'qcomment', 'programid']]
Obsdf_maglimit.reset_index(inplace=True, drop=True)
Obsdf_maglimit.expMJD.is_monotonic_increasing
Obsdf_maglimit.to_parquet(logspath+'ztf_obsfile_maglimit_new.parquet')



# Create the ztf survey plan for the simulations from Obsdf_maglimcat:

survey = pd.read_parquet(logspath+'ztf_obsfile_maglimcat_new.parquet')

survey.rename(columns={'expMJD':'mjd', 'filter':'band',
                                 'maglimcat':'maglim', 'fieldID':'fieldid'}, inplace=True)
survey['skynoise'] = 1/5 * 10**(-0.4*(survey.maglim - survey.zp))
survey = survey[(survey.programid==1)|(survey.programid==2)]

survey.to_pickle(logspath+'ztf_survey_12.pkl')
print(logspath+'ztf_survey_12.pkl   Saved :)')
