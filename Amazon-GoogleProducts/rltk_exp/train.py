import rltk
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

from featurize import featurize, get_document_frequency, featurize_record_pair, TRAIN_DOC_SIZE
from utils import impute_df, DATASET_DIR
from amazon_record import AmazonRecord
from google_record import GoogleRecord

ds_amzn = rltk.Dataset(reader=rltk.CSVReader(open(DATASET_DIR + 'Amazon.csv', encoding='latin-1')),
                record_class=AmazonRecord, adapter=rltk.MemoryAdapter())

ds_goog = rltk.Dataset(reader=rltk.CSVReader(open(DATASET_DIR + 'GoogleProducts.csv', encoding='latin-1')),
                record_class=GoogleRecord, adapter=rltk.MemoryAdapter())

def generate_features(gt_train):
    """
    Generate features from stratifed ground truth DataFrames

    Params:
        gt_train: (DataFrame) Df containing statified training data ids and labels
    """
    gt_train.label = gt_train.label.astype(int)
    df = pd.DataFrame()
    freq = get_document_frequency(DATA_DIR + 'train/corpus_freq.json', ds_amzn, ds_goog)
    for i in range(len(gt_train)):
        row = gt_train.iloc[i]
        r1 = ds_amzn.get_record(row.id1)
        r2 = ds_goog.get_record(row.id2)
        s = featurize_record_pair(r1, r2, freq, TRAIN_DOC_SIZE)
        s['label'] = row.label
        s['id1'] = row.id1
        s['id2'] = row.id2
        df = df.append(s, ignore_index=True)
    
    df.to_csv('dataset/train/features_train.csv')
    return df

def train(df, features, cls):
    """
    Run training pipeline

    Params:
        df: (DataFrame) training data
        features: (List) List of column names to take from dataframe
        cls: (Object) Sklearn classifier to train 
    """
    X_train = df[features].values
    y_train = df.label.values
    cls.fit(X_train,y_train)
    return cls

def plot_classifer_perfomance(train_df, yt):
    """
    Plot performance of different classifiers
    
    Params:
        train_df: (DataFrame) Training data
        yt: (np.array) List of true labels from test data 
    """
    classifiers = [ RandomForestClassifier(n_estimators=10),
            DecisionTreeClassifier(max_depth=5),
            MLPClassifier(alpha=1),
            svm.SVC(kernel='rbf', class_weight={1: 10}, probability=False),
            svm.SVC(kernel='linear', class_weight={1: 10}, probability=False),
            svm.SVC(kernel='rbf', class_weight={1: 100}, probability=False),
            svm.SVC(kernel='rbf', C=0.05, class_weight={1: 10}, probability=False)
            ]

    precision = []
    recall = []
    fscore = []

    for cls in classifiers:
        cls = train(train_df, features, cls)
        yp = cls.predict(xt)

        print(classification_report(yt,yp))
        pr, re, fs, s = precision_recall_fscore_support(yt, yp)
        precision.append(pr)
        recall.append(re)
        fscore.append(fs)

    x = [
        'Random Forest',
        'Decision tree classifier',
        'MLP classifer',
        'SVM rbf 1:10',
        'SVM linear 1:10',
        'SVM rbf 1:100',
        'SVM rbf c 0.05'
    ]
    precision = np.array(precision)
    recall = np.array(recall)
    fscore = np.array(fscore)
    plt.figure(figsize=(10,5))
    plt.plot(x,precision[:,0], label='Negative')
    plt.plot(x,precision[:,1], label='Positive')
    plt.xlabel('type of classifier')
    plt.ylabel('precision')
    plt.yticks(np.arange(0.0, 1.0, .2))
    plt.legend()
    plt.title("Precision")
    plt.savefig('precision.png', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(x,recall[:,0], label='Negative')
    plt.plot(x,recall[:,1], label='Positive')
    plt.ylabel('recall')
    plt.yticks(np.arange(0.0, 1.0, .2))
    plt.legend()
    plt.title("Recall")
    plt.savefig('recall.png', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(x,fscore[:,0], label='Negative')
    plt.plot(x,fscore[:,1], label='Positive')
    plt.ylabel('fscore')
    plt.yticks(np.arange(0.0, 1.0, .2))
    plt.legend()
    plt.title("fscore")
    plt.savefig('fscore.png', bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':

    try:
        df = pd.read_csv('train/features_train.csv')
    except FileNotFoundError:
        featurize(mode='train')
        df = pd.read_csv('train/features_train.csv')
    
    df = impute_df(df)

    features = df.columns.values.tolist()
    features.remove('id1')
    features.remove('id2')
    features.remove('label')

    try:
        dft = pd.read_csv('test/features_test.csv')
    except FileNotFoundError:
        featurize(mode='test')
        dft = pd.read_csv('test/features_test.csv')

    dft = impute_df(dft)
    xt = dft[features].values
    yt = dft.label.values

    cls = svm.SVC(kernel='rbf', class_weight={1: 10}, probability=True)
    cls = train(df, features, cls)
    yp = cls.predict(xt)
    dft['predicted'] = yp
    dft['proba'] = cls.predict_proba(xt)[:,1]

    print(classification_report(yt,yp))

    gt = rltk.GroundTruth()
    gt.load(DATASET_DIR+"Amzon_GoogleProducts_perfectMapping.csv")
    for i in range(len(dft)):
        row = dft.iloc[i]
        if row.label == 0:
            gt.add_negative(row.id1, row.id2)

    trial = rltk.Trial(gt)

    for i in range(len(dft)):
        row = dft.iloc[i]
        r1 = ds_amzn.get_record(row.id1)
        r2 = ds_goog.get_record(row.id2)
        proba = row.proba
        trial.add_result(r1, r2, row.predicted, proba)

    precision = []
    recall = []
    fscore = []
    threshold = 0.1
    
    x = list(range(1,100,15))
    for threshold in x:
        print('------------------------')
        trial.run_munkres(threshold=threshold/100)
        trial.evaluate()
        print('threshold:', threshold)
        print(trial.precision, trial.recall, trial.f_measure)
        precision.append(trial.precision)
        recall.append(trial.recall)
        fscore.append(trial.f_measure)
    
    plt.figure(figsize=(10,5))
    plt.plot(x,precision, label='Precision')
    plt.plot(x,recall, label='Recall')
    plt.plot(x,fscore, label='F_measure')
    plt.xlabel('number of strata')
    plt.ylabel('amount')
    plt.yticks(np.arange(0.0, 1.0, .2))
    plt.legend()
    plt.title("mukres results")
    plt.savefig('munkres.png', bbox_inches='tight')
   
       
    trial.evaluate()
