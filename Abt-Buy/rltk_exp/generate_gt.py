from feature_vector import *
from create_datasets import *
from sklearn.cluster import KMeans


gt = rltk.GroundTruth()
with open('../../datasets/Abt-Buy/abt_buy_perfectMapping.csv', encoding='latin-1') as f:
    for d in rltk.CSVReader(f):
        gt.add_positive(d['idAbt'], d['idBuy'])
gt_train, gt_test = gt.train_test_split(test_ratio=0.3)


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

gt_train.generate_stratified_negatives(ds_abt, ds_buy, classify, 10, range_in_gt=True, exclude_from=gt_test)
gt_train.save('gt_train_20180920.csv')

print('generating gt for testing')
gt_test.generate_all_negatives(ds_abt, ds_buy, range_in_gt=True, exclude_from=gt_train)
gt_test.save('gt_test_20180920.csv')
