from os import replace
import pandas as pd 
import numpy as np 
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.metrics import classification_report
import json
import scipy

df = pd.read_csv('eval_valid.csv')
# df = df.head(10)
preds = np.array(df['Prediction'].tolist())
label = np.array(df['Label'].tolist())

# index = np.arange(0, len(df))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m+h, m-h

def process(cur):
    np.random.seed(cur)
    idxs = np.random.choice(len(df), len(df), replace=True)
    # print(idxs)
    new_p = preds[idxs]
    new_l = label[idxs]
    ret = classification_report(new_l, new_p, output_dict=True)
    # print(ret)
    return ret

all_ret = {}
all_metric = {}
with Pool(processes=8) as p:
    max_ = 5000
    with tqdm(total=max_) as pbar:
        for i, (metric) in enumerate(
            p.imap_unordered(process, range(max_))
        ):
            for item in ['0', '1', '2', '3', 'macro avg']:
                if not item in all_metric:
                    all_metric[item] = {}
                for name in metric[item]:
                    if name != "support":
                        if not name in all_metric[item]:
                            all_metric[item][name] = []
                        all_metric[item][name].append(metric[item][name])
            all_ret[i] = metric
            pbar.update()

with open('data.json', 'w') as outfile:
    json.dump(all_ret, outfile)

for item in ['0', '1', '2', '3', 'macro avg']:
    for name in ['precision', 'recall', 'f1-score']:
        tmp = np.sort(all_metric[item][name])
        lower = int(5000 * 0.025)
        upper = int(5000 * 0.975)
        print(item, name, tmp[lower], tmp[upper])
        # ret = mean_confidence_interval(tmp)
        # print(item, name, ret)

# print(all_metric['0']['precision'])
