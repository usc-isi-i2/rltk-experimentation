import rltk
from sklearn.cluster import KMeans
import random
import pandas as pd
import json
import numpy as np

from featurize import featurize_record_pair
from amazon_record import AmazonRecord
from google_record import GoogleRecord
from train import *
from utils import impute_df, impute_s, DATASET_DIR

"""
Implement stratified sampling on the train dataset and save to disk
"""

DATA_DIR = 'train/'
ds_amzn = rltk.Dataset(reader=rltk.CSVReader(open(DATA_DIR + 'Amazon.csv', encoding='latin-1')),
                   record_class=AmazonRecord, adapter=rltk.MemoryAdapter())

ds_goog = rltk.Dataset(reader=rltk.CSVReader(open(DATA_DIR + 'GoogleProducts.csv', encoding='latin-1')),
                   record_class=GoogleRecord, adapter=rltk.MemoryAdapter())

train_df = pd.read_csv(DATA_DIR+'features_train.csv')
train_df = impute_df(train_df)
train_df = train_df[train_df.label==1]

features = train_df.columns.values.tolist()
features.remove('id1')
features.remove('id2')
features.remove('label')

X_train = train_df[features].values
y_train = train_df.label.values

kmeans_model = KMeans(n_clusters=10, random_state=0).fit(X_train)

dft = pd.read_csv('test/features_test.csv')
dft = impute_df(dft)
xt = dft[features].values
yt = dft.label.values
gt = rltk.GroundTruth()
gt.load(DATASET_DIR+"Amzon_GoogleProducts_perfectMapping.csv")
for i in range(len(dft)):
    row = dft.iloc[i]
    if row.label == 0:
        gt.add_negative(row.id1, row.id2)


with open('train/corpus_freq.json') as f:
    train_freq = json.load(f)

def classify(r1, r2):
    v = featurize_record_pair(r1, r2, train_freq, 3444)
    v = v.drop(['id1','id2','label'])
    v = impute_s(v)
    cluster_id = kmeans_model.predict([v])[0]
    return cluster_id

precision = []
recall = []
fscore = []
strata = list(range(15,150,25))
for n_strata in strata:
    print(n_strata)
    gt_train = rltk.GroundTruth()
    for i in range(len(train_df)):
        row = train_df.iloc[i]
        gt_train.add_positive(row.id1, row.id2)
    gt_train.generate_stratified_negatives(ds_amzn, ds_goog, classify, n_strata, range_in_gt=True)
    gt_train.save('train/stratified_train_gt_'+str(n_strata)+'.csv')

