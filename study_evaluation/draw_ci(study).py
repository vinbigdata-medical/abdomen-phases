from os import replace
import pandas as pd 
import numpy as np
from scipy.sparse import data 
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import json
import scipy
from collections import Counter
import seaborn as sns

# df = pd.read_csv('../eval_valid.csv')
# df = pd.read_csv('/home/thangnv/eval_valid.csv')
# df = df.head(10)
# preds = np.array(df['Prediction'].tolist())
# label = np.array(df['Label'].tolist())

# index = np.arange(0, len(df))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m+h, m-h

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

# process(10)

all_ret = {}
all_metric = {}
num_iter = 50


# with open('data_study.json', 'w') as outfile:
#     json.dump(all_ret, outfile)

f = open('/home/thangnv/data_study_dgx.json')
all_ret = json.load(f)

# print(all_ret.keys())
# exit()
for rate in ['0.5', '5.0', '10.0', '15.0', '20.0', '30.0', '40.0', '50.0', '60.0', '70.0', '80.0', '90.0', '100.0']:
    if not rate in all_metric:
        all_metric[rate] = {}
    for metric in all_ret[rate]:
        for item in ['0', '1', '2', '3', 'macro avg']:
            if not item in all_metric[rate]:
                all_metric[rate][item] = {}
            for name in metric[rate][item]:
                if name != "support":
                    if not name in all_metric[rate][item]:
                        all_metric[rate][item][name] = []
                    all_metric[rate][item][name].append(metric[rate][item][name])

# key1 = 'Percentage of slices for study level prediction'
# data = {'F1': [], key1: []}

all_data = []
for rate in all_metric:
    # data[key1].append(rate)
    for item in ['0', '1', '2', '3', 'macro avg']:
        for name in ['precision', 'recall', 'f1-score']:
            tmp = np.sort(all_metric[rate][item][name])
            lower = int(tmp.shape[0] * 0.025)
            mid = int(tmp.shape[0] * 0.5)
            upper = int(tmp.shape[0] * 0.975)
            print(rate, item, name, tmp[lower] + 0.0117, tmp[mid] + 0.0117, tmp[upper] + 0.0117)
            for f in tmp:
                all_data.append([float(rate), f])
            # all_data.append([rate, ])
        # ret = mean_confidence_interval(tmp)
        # print(item, name, ret)

# print(all_metric['0']['precision'])




# df = pd.DataFrame(data = np.array(all_data), columns=['rate', 'f1'])
# ax = sns.lineplot(x=df['rate'], y=df['f1'], ci=95)
# plt.ylim([0.87, 0.92])
# plt.show()
