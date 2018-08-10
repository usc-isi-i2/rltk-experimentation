import rltk
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




# # load gt
gt_train = rltk.GroundTruth()
gt_train.load('gt_train_new.csv')
gt_test = rltk.GroundTruth()
gt_test.load('gt_test_new.csv')
# gt_test.load('gt_full.csv')

# generate ground truth
# print('construct kmeans input')
# X_km = []
# for r_abt, r_buy in rltk.get_record_pairs(ds_abt, ds_buy):
#     X_km.append(generate_feature_vector(r_abt, r_buy))
# kmeans_model = KMeans(n_clusters=10, random_state=0).fit(X_km)
# with open('kmeans_model.pkl', 'wb') as f:
#     pickle.dump(kmeans_model, f)

#
# with open('kmeans_model.pkl', 'rb') as f:
#     kmeans_model = pickle.load(f)
#
#
# def classify(r_abt, r_buy):
#     v = generate_feature_vector(r_abt, r_buy)
#     cluster_id = kmeans_model.predict([v])[0]
#     return cluster_id
#
# print('construct gt')
# for i in range(1, 6):
#     num_of_negatives = 1098 * (2 ** i)
#     gt = rltk.GroundTruth()
#     reader = rltk.CSVReader(open('../../datasets/Abt-Buy/abt_buy_perfectMapping.csv', encoding='latin-1'))
#     for r in reader:
#         id_abt = r['idAbt']
#         id_buy = r['idBuy']
#         gt.add_positive(id_abt, id_buy)
#
#     print('generate negatives: {}'.format(num_of_negatives))
#     gt.generate_stratified_negatives(ds_abt, ds_buy, classify, 10, num_of_negatives=num_of_negatives)
#     gt_train, gt_test = gt.train_test_split()
#     gt_train.save('gt_train_strata_{}.csv'.format(num_of_negatives))
#     gt_test.save('gt_test_strata_{}.csv'.format(num_of_negatives))

# gt = rltk.GroundTruth()
# reader = rltk.CSVReader(open('../../datasets/Abt-Buy/abt_buy_perfectMapping.csv', encoding='latin-1'))
# for r in reader:
#     id_abt = r['idAbt']
#     id_buy = r['idBuy']
#     gt.add_positive(id_abt, id_buy)
#
# gt.generate_all_negatives(ds_abt, ds_buy)
# gt_train, gt_test = gt.train_test_split(test_ratio=0.95)
# gt_train.save('gt_train_strata_dot05.csv')
# gt_test.save('gt_test_strata_dot05.csv')
# exit()
#
# X, y = [], []
# train_pairs = rltk.get_record_pairs(ds_abt, ds_buy, ground_truth=gt_train)
# for r_abt, r_buy in train_pairs:
#     v = generate_feature_vector(r_abt, r_buy)
#     X.append(v)
#     y.append(gt_train.get_label(r_abt.id, r_buy.id))

# clf = svm.SVC()
# clf = DecisionTreeClassifier()
# clf = RandomForestClassifier()
# clf.fit(X, y)
#
# with open('model.pkl', 'wb') as f:
#     pickle.dump(clf, f)
#
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

# trial = rltk.Trial(ground_truth=gt_test)
# test_pairs = rltk.get_record_pairs(ds_abt, ds_buy, ground_truth=gt_test)

trial_list = []
granularity = 10
# for g in range(granularity + 1):
g = 2.9
threshold = float(1) / granularity * g
trial = rltk.Trial(ground_truth=gt_test, threshold=threshold)
for r_abt, r_buy in rltk.get_record_pairs(ds_abt, ds_buy, ground_truth=gt_test):
    # ml
    v = generate_feature_vector(r_abt, r_buy)
    vv = clf.predict_proba([v])
    trial.add_result(r_abt, r_buy,
                     vv[0][1] >= threshold,
                     confidence=vv[0][1],
                     feature_vector=v)
# trial.evaluate()
# trial_list.append(trial)
#     # break
#
# trial = trial_list[0]

# for t in trial_list:
#     print(t.precision, t.recall, t.threshold)
# eva = rltk.Evaluation(trial_list)
# eva.plot_precision_recall().show()
# p = eva.plot([
#     {
#         'x': 'threshold',
#         'y': 'precision',
#         'label': 'precision'
#     },
#     {
#         'x': 'threshold',
#         'y': 'recall',
#         'label': 'recall'
#     }
# ])
# p.xlabel("threshold")
# p.legend(loc="bottom right")
# p.show()


trial.run_munkres()
trial.evaluate()
print(trial.true_positives, trial.false_positives, trial.true_negatives, trial.false_negatives,
      trial.precision, trial.recall, trial.f_measure)

# print(len(trial.true_positives_list))
# print(len(trial.false_positives_list))
# print(len(trial.true_negatives_list))
# print(len(trial.false_negatives_list))
# #
# # for result in trial.false_negatives_list:
# #     print('-------------')
# #     print('', result.record1.id, result.record1.name, '\tbrand:', result.record1.brand_cleaned, '\tmodel:',
# #           result.record1.model_cleaned,
# #           '\n',
# #           result.record2.id, result.record2.name, '\tbrand:', result.record2.brand_cleaned, '\tmodel:',
# #           result.record2.model_cleaned,
# #           '\n',
# #           'confidence:', result.confidence)
# #
# pd = trial.generate_dataframe(trial.false_negatives_list,
#                               record1_columns=['id'], record2_columns=['id'],
#                               result_columns=['is_positive', 'confidence'])
# df = trial.generate_dataframe(trial.true_positives_list)
# pd.set_option('display.max_colwidth', -1)
# df.to_csv('report_tp.csv')
# df = trial.generate_dataframe(trial.false_positives_list)
# pd.set_option('display.max_colwidth', -1)
# df.to_csv('report_fp.csv')
# df = trial.generate_dataframe(trial.true_negatives_list)
# pd.set_option('display.max_colwidth', -1)
# df.to_csv('report_tn.csv')
# df = trial.generate_dataframe(trial.false_negatives_list)
# pd.set_option('display.max_colwidth', -1)
# df.to_csv('report_fn.csv')