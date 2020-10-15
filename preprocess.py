#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anonymous
"""
import pickle
import argparse
import pandas as pd
import numpy as np
import progressbar
from datetime import datetime, timedelta

from utilities import *

##### Labels #####
def make_labels(opts):
    icu_details = pd.read_csv(opts.path_views + '/icustay_detail.csv')
    #apply exclusion criterias
    icu_details = icu_details[(icu_details.age>=18)&(icu_details.los_hospital>=1)&(icu_details.los_icu>=1)]
    sepsis3 = pd.read_csv(opts.path_sepsis3)
    subj = list(set(icu_details.subject_id.tolist()))

    #make labels
    dct = {}
    print("="*80)
    print("Generating Labels...".center(80))
    print("="*80)
    for sample in progressbar.progressbar(range(len(subj))):
        s = subj[sample]
        lst = icu_details[icu_details.subject_id==s].hadm_id.tolist()
        
        times = [(pd.to_datetime(icu_details[icu_details.hadm_id==i].admittime.values[0]),
                  pd.to_datetime(icu_details[icu_details.hadm_id==i].dischtime.values[0]), i) for i in lst]
        times = list(set(times))
        times = sorted(times, key= lambda x: x[0])
    
        morts = [(icu_details[icu_details.hadm_id == h[-1]].hospital_expire_flag.values[0], h[-1]) for h in times]
        sepsis = [h[-1] for h in times if 1 in sepsis3[sepsis3.hadm_id == h[-1]]['sepsis-3'].values]
        mortalities = [m[-1] for m in morts if m[0] ==1]

        if len(mortalities)==1:
            hadm = mortalities[0]  #pick the mortality stay if no readmission
        elif len(sepsis) >=1:
            hadm = sepsis[-1]
        else:
            lengths = [(t[1] - t[0],t[-1]) for t in times]
            hadm = sorted(lengths, key = lambda x: x[0])[-1][-1]    #pick the longest stay if no readmit and no deaths.
        
        dct[s] = {'hadm_id': hadm, 
                   'mort': icu_details[icu_details.hadm_id == hadm].hospital_expire_flag.values[0],
                   'sepsis': sepsis3[sepsis3.hadm_id == hadm]['sepsis-3'].values[0]}
    return dct


##### Diagnosis Pivot Table ######
def pivot_icd(subj):
    '''subj: list of cohort subject_ids'''
    df = pd.read_csv(path_tables + '/diagnoses_icd.csv')
    #icd names
    icd_names = pd.read_csv(path_tables + '/d_icd_diagnoses.csv')
    #make dictionary of icd9 codes
    dct = {}
    for i in progressbar.progressbar(range(len(subj))):
        s = subj[i]
        dictionary = df[(df.subject_id == s)][['hadm_id', 'icd9_code']].groupby('hadm_id')['icd9_code'].apply(list).to_dict()
        dictionary = dict([(k,v ) for k,v in dictionary.items()])
        dct[s] = dictionary
    lengths = [dct[i].values() for i in dct.keys()]
    lengths = flatten(lengths)
    lengths = flatten(lengths)
    unique, counts = np.unique(lengths, return_counts=True)
    #frequency dictionary
    dct_freq = dict(zip(unique, counts))
    items = sorted(dct_freq.items(), key = lambda x: x[1], reverse = True)
    ## add names ##
    common = list(set(icd_names.icd9_code).intersection([i[0] for i in items]))
    common = icd_names[icd_names.icd9_code.isin(common)]
    common = common[['icd9_code', 'short_title']].groupby('icd9_code')['short_title'].apply(list).to_dict()
    dct_freq = []
    for idx, count in items:
        if idx in common.keys():
            dct_freq.append((idx, common[idx][0], count))
    return dct, dct_freq

#### Features ####
def get_features(patients):
    '''patients: {subject_id: hadm_id}'''
    p_bg = pd.read_csv(path_views + '/pivoted_bg.csv')
    p_vital= pd.read_csv(path_views + '/pivoted_vital.csv')
    p_lab = pd.read_csv(path_views + '/pivoted_lab.csv')
    cohort = pd.read_csv(path_views + '/icustay_detail.csv')
    ## Exclusion criteria ##
    cohort = cohort[cohort.subject_id.isin(patients.keys())&(cohort.hadm_id.isin(patients.values()))]

    ## hourly binning ##
    p_bg.charttime = pd.to_datetime(p_bg.charttime)
    p_bg = p_bg.dropna(subset=['hadm_id'])
    p_vital.charttime = pd.to_datetime(p_vital.charttime)
    p_vital = p_vital.dropna(subset=['icustay_id'])
    p_lab.charttime = pd.to_datetime(p_lab.charttime)
    p_lab = p_lab.dropna(subset=['hadm_id'])
    
    ## initialize icustays dict ##
    dct_bins = {}
    lst= sorted(list(set(cohort.hadm_id)))
    hadm_dct = dict([(h, cohort[cohort['hadm_id']==h].subject_id.values[0]) for h in lst])
    
    icu_hadm = dict([(h, cohort[cohort.hadm_id == h].icustay_id.tolist()) for h in lst])
    icu_dct = {}
    for key,val in icu_hadm.items():
        for v in val:
            icu_dct[v] = key
    icustays = sorted(icu_dct.keys())
    
    ref_ranges = [83.757076260234783, 118.82208983706279, 61.950770137747298, 
                  36.73772, 18.567563025210085, 96.941729323308266, 90.0,
                  4.5, 12.5, 0.89999999999999991, 140.5, 25.0, 275.0, 1.61,
                  4.25, 1.140, 7.4000000000000004, 39.0,1.5]
    dfs = [p_vital, p_lab, p_bg]
    lsts = [icustays, lst, lst]
    cols = [['heartrate', 'sysbp', 'diasbp', 'tempc', 'resprate', 'spo2', 'glucose'],
                   ['albumin', 'bun','creatinine', 'sodium', 'bicarbonate', 'platelet', 'inr'], 
                   ['potassium', 'calcium', 'ph', 'pco2', 'lactate']]
    
    ## initialize features by filtered hadm ##
    features = {}
    subj = sorted(set(cohort.subject_id))

    print("Initializing Timesteps..." )
    print("........")
    for i in progressbar.progressbar(range(len(subj))):
        s = subj[i]
        hadm = patients[s]
        timesteps = [pd.to_datetime(datetime.strptime(cohort[cohort.hadm_id==hadm].admittime.values[0], 
                                          '%Y-%m-%d %H:%M:%S') + timedelta(hours=hr)) for hr in range(48)]
        timesteps = [tt.replace(microsecond=0,second=0,minute=0) for tt in timesteps]
        features[hadm] = {}
        for t in timesteps:
            features[hadm][t] = {}

    print()
    print("Eliminating samples with too few timesteps...")
    ## eliminate low time-step samples ##
    lst = []
    #initialize timestamps with vital signs
    for j in progressbar.progressbar(range(len(icustays))):
        h = icustays[j]
        if icu_dct[h] in features.keys():
            timesteps = [i for i in p_vital[p_vital['icustay_id']==h].set_index('charttime').resample('H').first().index.tolist() if i <= max(features[icu_dct[h]].keys())]
            if len(timesteps) >= 6:
                lst.append(icu_dct[h])

    #get timestamps for labs
    lst2 = []
    for j in progressbar.progressbar(range(len(lst))):
        h = lst[j]
        timesteps = [i for i in p_lab[p_lab['hadm_id']==h].set_index('charttime').resample('H').first().index.tolist() if i <= max(features[h].keys())]
        if len(timesteps)>=1:
            lst2.append(h)
    lst = lst2; del lst2
    #update icustays list and features
    features = dict([(k,v) for k,v in features.items() if k in lst])
    icu_hadm = dict([(h, cohort[cohort.hadm_id == h].icustay_id.tolist()) for h in lst])
    icu_dct = {}
    for key,val in icu_hadm.items():
        for v in val:
            icu_dct[v] = key
    icustays = sorted(icu_dct.keys())

    print()
    print("="*80)
    print("Generating Timeseries Features")
    print("="*80)
    print()
    lsts = [icustays, lst, lst]
    feature_index=0
    for idx in range(len(dfs)):
        for c in cols[idx]:
            top5 = dfs[idx][c].quantile(.95) 
            bot5 = dfs[idx][c].quantile(.05) 
            print('{0}: {1}'.format( c, ref_ranges[feature_index]))
            print()
            #for each admission, for each hourly bin, construct feature vector
            for i in progressbar.progressbar(range(len(lsts[idx]))):
                h = lsts[idx][i]
                if len(lst) == len(lsts[idx]):
                    s = dfs[idx][dfs[idx]['hadm_id']==h].set_index('charttime')[c]
                else:
                    s =  dfs[idx][dfs[idx]['icustay_id']==h].set_index('charttime')[c]
                    h = icu_dct[h]
                
                s = s.interpolate(limit_direction = 'both', limit_area = 'inside')
                s = s.fillna(ref_ranges[feature_index])
                time_range= sorted(features[h].keys())
                s = s.loc[time_range[0]: time_range[-1]]
                if len(s)>0:
                    s= s.resample('H').ohlc()['close'].interpolate(limit_direction='both')

                    s = s.reindex(pd.to_datetime(time_range))
                    s = s.interpolate()
                    s = s.fillna(ref_ranges[feature_index])
                    s[s>=top5] = top5
                    s[s<=bot5] = bot5
                    s = dict([(key,val) for key,val in s.items() if key <= max(features[h].keys())])
                    times = sorted(s.keys())
                    for t in time_range:
                        if t < times[0]:
                            features[h][t][c] = s[times[0]]
                        elif t in times:
                            features[h][t][c] = s[t]
                        elif t not in s.keys():
                            curr = find_nearest(times, t)
                            features[h][t][c] = s[curr]
                            s[t] = s[curr]
                        else:
                            print(times, t)
                    if pd.isnull(list(s.values())).any():
                        print(s)
                else:
                    for t in sorted(features[h].keys()):
                        features[h][t][c] = ref_ranges[feature_index]
            feature_index+=1
    return features

#### Preprocessing ####
def preprocess(features, labels):
    '''pre: features and labels
    post: X = [[x1, ... xT]_1, ...], y= [(mort, readm, los, dx)] '''
    from sklearn.preprocessing import MinMaxScaler
    subj = list(set(labels.keys()))   
    hadm = list(set(features.keys()))
    col_dict = dict ([(v,k) for k,v in enumerate(features[hadm[0]][list(features[hadm[0]].keys())[0]].keys())])
    cols = sorted(col_dict.keys())
    items = []
    for i in progressbar.progressbar(range(len( subj ) ) ):
        s = subj[i]
        h = labels[s]['hadm_id']
        if h in hadm:
            x = np.zeros((len(features[h].keys()), len(col_dict)))
            for index in range(len(sorted(features[h].keys()))):
                t = sorted(features[h].keys())[index]
                x[index, [col_dict[k] for k in cols]] = [features[h][t][k] for k in cols]
            mort = labels[s]['mort']
            sepsis = labels[s]['sepsis']
            y = (mort, sepsis)
            items.append((x, y))
    X, y = zip(*items)
    X, y = np.array(list(X)), list(y)
    #normalize each feature to [0,1]
    for i in range(len(X[0,0,:])):
        #scale X
        scaler = MinMaxScaler()
        x_row = scaler.fit_transform(X[:,:,i])
        X[:,:,i] = x_row
    return X, y


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Training hyper-parameters
    parser.add_argument('--path_tables', type=str, default='~/local_mimic/tables',
                        help='Path to the original MIMIC-III Tables.')
    parser.add_argument('--path_sepsis3', type=str, default='~/local_mimic/sepsis3-data/sepsis3-df-no-exclusions.csv',
                        help='Path to the Sepsis-3 Table.')
    parser.add_argument('--path_views', type=str, default='~/local_mimic/views',
                        help='Path to View tables from MIMIC-III cookbook.')
    parser.add_argument('--path_save', type=str, default='~/tmp',
                        help='Set the directory to store features, labels and such.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    
    path_tables = opts.path_tables
    path_views = opts.path_views
    path_save = opts.path_save

    if (path_tables and path_views and path_save):
        create_dir_if_not_exists(path_save)
    
        labels = make_labels(opts)
        with open(path_save + '/labels', 'wb') as f:
            pickle.dump(labels, f)
        print("Saving labels ..." )
        print("........")
        print("Done!")
        print("Constructing Feature Space ..... ")
        print("........")
        patients = dict([(k, labels[k]['hadm_id']) for k in labels.keys()])
        features = get_features(patients)
        patients = dict([(k,v) for k,v in patients.items() if v in features.keys()])
        with open(path_save + '/features', 'wb') as f:
            pickle.dump(features,f)
        print("Done!")
        print()
        print("Preprocessing to construct X and y.")
        X, y= preprocess(features, labels)
        np.save(path_save + '/X', X)
        with open(path_save+'/y', 'wb') as f:
            pickle.dump(y, f)
        print("Done!")
    else:
        print("Make sure you have the MIMIC-III Tables and Views.")
        print("See Requirements page.")
        