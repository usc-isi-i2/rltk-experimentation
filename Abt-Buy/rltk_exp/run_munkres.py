from create_datasets import *
from feature_vector import *
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import numpy as np
import pickle
import pandas as pd
import collections
import json


# load gt
gt_train = rltk.GroundTruth()
gt_train.load('gt_train_20180810.csv')
gt_test = rltk.GroundTruth()
gt_test.load('gt_test_20180810.csv')

# ------------------------------
# train model
X, y = [], []
train_pairs = rltk.get_record_pairs(ds_abt, ds_buy, ground_truth=gt_train)
for r_abt, r_buy in train_pairs:
    v = generate_feature_vector(r_abt, r_buy)
    X.append(v)
    y.append(gt_train.get_label(r_abt.id, r_buy.id))

clf = svm.SVC(probability=True)
# clf = DecisionTreeClassifier()
# clf = RandomForestClassifier()
clf.fit(X, y)

# with open('model.pkl', 'wb') as f:
#     pickle.dump(clf, f)
# exit()
#
# with open('model.pkl', 'rb') as f:
#     clf = pickle.load(f)

# ------------------------------
# generate fv

# f = open('features.jl', 'w')
# trial = rltk.Trial(ground_truth=gt_test)
# for r_abt, r_buy in rltk.get_record_pairs(ds_abt, ds_buy, ground_truth=gt_test):
#     # ml
#     v = generate_feature_vector(r_abt, r_buy)
#     vv = clf.predict_proba([v])[0][1]
#     trial.add_result(r_abt, r_buy, vv > 0.3,
#                      confidence=vv,
#                      feature_vector=v)
#     f.write(json.dumps({'id1':r_abt.id, 'id2':r_buy.id, 'fv':v, 'proba':vv}) + '\n')
# f.close()
# exit()

# f = open('features.jl')
# trial = rltk.Trial(ground_truth=gt_test)
# for r_abt, r_buy in rltk.get_record_pairs(ds_abt, ds_buy, ground_truth=gt_test):
#     # ml
#     obj = json.loads(f.readline())
#     v = obj['fv']
#     vv = obj['proba']
#     trial.add_result(r_abt, r_buy, vv > 0.3,
#                      confidence=vv,
#                      feature_vector=v)
# f.close()

trial = rltk.Trial(ground_truth=gt_test)
for r_abt, r_buy in rltk.get_record_pairs(ds_abt, ds_buy, ground_truth=gt_test):
    # ml
    v = generate_feature_vector(r_abt, r_buy)
    vv = clf.predict_proba([v])[0][1]
    trial.add_result(r_abt, r_buy, vv > 0.3,
                     confidence=vv,
                     feature_vector=v)

# ------------------------------
# evaluation

for threshold in [x * 0.1 for x in range(0, 10)]:
    print('------------------------')
    trial.run_munkres(threshold=threshold)
    trial.evaluate()
    print('threshold:', threshold)
    print(trial.true_positives, trial.false_positives, trial.true_negatives, trial.false_negatives,
          trial.precision, trial.recall, trial.f_measure)
    print('tp:', len(trial.true_positives_list), end=' ')
    print('fp:', len(trial.false_positives_list), end=' ')
    print('tn:', len(trial.true_negatives_list), end=' ')
    print('fn:', len(trial.false_negatives_list))
