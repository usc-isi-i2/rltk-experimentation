from feature_vector import *
from create_datasets import *
from sklearn.cluster import KMeans
import random


gt_train = rltk.GroundTruth()
gt_test = rltk.GroundTruth()

abt_train_id_set = set([])
buy_train_id_set = set([])
abt_test_id_set = set([])
buy_test_id_set = set([])
perfect_pairs = []
train_ratio = 0.7
reader = rltk.CSVReader(open('../../datasets/Abt-Buy/abt_buy_perfectMapping.csv', encoding='latin-1'))
for r in reader:
    id_abt = r['idAbt']
    id_buy = r['idBuy']
    perfect_pairs.append((id_abt, id_buy))
    abt_test_id_set.add(id_abt)
    buy_test_id_set.add(id_buy)

print('generating gt for training')
train_selected = random.sample(perfect_pairs, int(len(perfect_pairs) * 0.7))
for id_abt, id_buy in train_selected:
    gt_train.add_positive(id_abt, id_buy)
    abt_train_id_set.add(id_abt)
    buy_train_id_set.add(id_buy)
    abt_test_id_set.discard(id_abt)
    buy_test_id_set.discard(id_buy)


X_km = []
for id_abt, id_buy, _ in gt_train:
    r_abt = ds_abt.get_record(id_abt)
    r_buy = ds_buy.get_record(id_buy)
    X_km.append(generate_feature_vector(r_abt, r_buy))
kmeans_model = KMeans(n_clusters=10, random_state=0).fit(X_km)


def classify(r_abt, r_buy):
    v = generate_feature_vector(r_abt, r_buy)
    cluster_id = kmeans_model.predict([v])[0]
    return cluster_id

gt_train.generate_stratified_negatives(ds_abt, ds_buy, classify, 10, range_in_gt=True)
gt_train.save('gt_train_20180810.csv')

print('generating gt for testing')
for id_abt, id_buy in perfect_pairs:
    if id_abt in abt_test_id_set and id_buy in buy_test_id_set:
        gt_test.add_positive(id_abt, id_buy)
gt_test.generate_all_negatives(ds_abt, ds_buy, range_in_gt=True)
gt_test.save('gt_test_20180810.csv')
