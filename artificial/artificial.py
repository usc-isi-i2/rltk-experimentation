import os
import json
import pandas as pd
import rltk
from datetime import datetime, timedelta
from random import randrange


def random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)


selected = dict()
df_gt = pd.read_csv('../datasets/DBLP-Scholar/DBLP-Scholar_perfectMapping.csv', encoding='latin-1').sample(100)
df_gt = df_gt[df_gt['idDBLP'].map(len) < 50]
df_gt = df_gt[df_gt['idScholar'].map(len) < 50]
for d in rltk.DataFrameReader(df_gt):
    date = random_date(datetime(2018, 12, 20), datetime(2018, 12, 30)).date()
    selected[d['idDBLP']] = date
    selected[d['idScholar']] = date
df_gt.to_csv('dblp_scholar_gt.csv', index=False)

df_dblp = pd.read_csv('../datasets/DBLP-Scholar/DBLP1.csv', encoding='latin-1')
df_dblp_out = {
    'id': [],
    'names': [],
    'date': []
}
for _, row in df_dblp.iterrows():
    # print(row['id'], row['authors'], row['year'])
    if row['id'] in selected:
        if not isinstance(row['authors'], str):
            continue
        df_dblp_out['id'].append(row['id'])
        df_dblp_out['names'].append(row['authors'])
        df_dblp_out['date'].append(selected[row['id']])
df_dblp_out = pd.DataFrame(data=df_dblp_out)
df_dblp_out.to_csv('dblp.csv', index=False)

df_scholar = pd.read_csv('../datasets/DBLP-Scholar/Scholar.csv', encoding='latin-1')
df_scholar_out = {
    'id': [],
    'names': [],
    'date': []
}
for _, row in df_scholar.iterrows():
    if row['id'] in selected:
        if not isinstance(row['authors'], str):
            continue
        df_scholar_out['id'].append(row['id'])
        df_scholar_out['names'].append(row['authors'])
        df_scholar_out['date'].append(selected[row['id']].strftime('%d, %b %Y'))
df_scholar_out = pd.DataFrame(data=df_scholar_out)
with open('scholar.jl', 'w') as f:
    for r in json.loads(df_scholar_out.to_json(orient='records')):
        f.write(json.dumps(r) + '\n')

