from os import replace
import pandas as pd 
import numpy as np 
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.metrics import classification_report
import json
import scipy
from collections import Counter

# df = pd.read_csv('../eval_valid.csv')
df = pd.read_csv('eval_valid.csv')
# df = df.head(10)
# preds = np.array(df['Prediction'].tolist())
# label = np.array(df['Label'].tolist())

# index = np.arange(0, len(df))

def process(info):
    seed = info[0]
    rate = info[1]
    all_preds = []
    all_labels = []
    np.random.seed(seed)
    for (studyuid, seriesuid), tmp_df in df.groupby(['Study_ID', 'SeriesNumber']):
        tmp_df = tmp_df.reset_index(drop=True)
        preds = np.array(tmp_df['Prediction'].tolist())
        label = np.array(tmp_df['Label'].tolist())
        if len(tmp_df['Label'].unique().tolist()) > 1:
            print("zzzzzzzzzzzzz")
        num_pick = int(len(tmp_df) * (rate / 100.0))
        if num_pick == 0:
            num_pick = 1
        if num_pick > len(tmp_df):
            print(rate, len(tmp_df), num_pick)
        # num_pick = 30
        idxs = np.random.choice(len(tmp_df), num_pick, replace=False)  
        new_p = preds[idxs]  
        new_l = label[idxs]
        series_p = Counter(new_p).most_common(1)[0][0]
        series_l = Counter(new_l).most_common(1)[0][0]
        all_preds.append(series_p)
        all_labels.append(series_l)

    # all_preds = np.array(all_preds)    
    # all_labels = np.array(all_labels)
    ret = classification_report(all_labels, all_preds, output_dict=True)
    # print(classification_report(all_labels, all_preds))
    # print(ret)
    return {rate: ret}

# print(process([10, 45]))

all_ret = {}
all_metric = {}
num_iter = 500
jobs = []
rate = np.array([0.1, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
for rid in range(rate.shape[0]):
    for idx in range(num_iter):
        jobs.append([rid * num_iter + idx, rate[rid]])

print("total", len(jobs))
with Pool(processes=16) as p:
    
    with tqdm(total=len(jobs)) as pbar:
        for i, (metric) in enumerate(
            p.imap_unordered(process, jobs)
        ):
            rate = list(metric.keys())[0]
            if not rate in all_ret:
                all_ret[rate] = []
            if not rate in all_metric:
                all_metric[rate] = {}
            for item in ['0', '1', '2', '3', 'macro avg']:
                if not item in all_metric[rate]:
                    all_metric[rate][item] = {}
                for name in metric[rate][item]:
                    if name != "support":
                        if not name in all_metric[rate][item]:
                            all_metric[rate][item][name] = []
                        all_metric[rate][item][name].append(metric[rate][item][name])
            all_ret[rate].append(metric)
            pbar.update()

with open('data_study.json', 'w') as outfile:
    json.dump(all_ret, outfile)

for rate in all_metric:
    for item in ['0', '1', '2', '3', 'macro avg']:
        for name in ['precision', 'recall', 'f1-score']:
            tmp = np.sort(all_metric[rate][item][name])
            lower = int(num_iter * 0.025)
            upper = int(num_iter * 0.975)
            mid = int(num_iter * 0.5)
            print(rate, item, name, tmp[lower], tmp[mid], tmp[upper])
        
