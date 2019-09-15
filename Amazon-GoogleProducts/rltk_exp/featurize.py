import rltk
import sys
import pandas as pd
import os
import random
from collections import Counter
import json

from amazon_record import AmazonRecord
from google_record import GoogleRecord
from utils import tokenize, DATASET_DIR

def tokenizer(record):
    """
    Tokenizer for inverted index blocking

    Params:
        record: (rltk.Record)
    """
    return record.name_tokenized

def featurize_record_pair(r1, r2, freq, doc_size):
    """
    Featurize a record pair and return a Series of the feature vectors

    Params:
        r1: (rltk.Record) record 1
        r2: (rltk.Record) record 2
        freq: (Dict) corpus frequency
        doc_size: (int) total size of dataset
    """
    fv = pd.Series()
    fv['id1'] = r1.id
    fv['id2'] = r2.id

    if gt.is_member(r1.id, r2.id):
        fv['label'] = 1
    else:
        fv['label'] = 0

    if (r1.manufacturer == '' or None) or (r2.manufacturer == '' or None):
        fv['manufacturer_jaro_winkler'] = None 
        fv['manufacturer_levenshtien'] = None
        fv['manufacturer_jaccard'] = None
    else:
        fv['manufacturer_jaro_winkler'] = rltk.jaro_winkler_similarity(r1.manufacturer, r2.manufacturer)
        fv['manufacturer_levenshtien'] = rltk.levenshtein_similarity(r1.manufacturer, r2.manufacturer)
        fv['manufacturer_jaccard'] = rltk.jaccard_index_similarity(set(tokenize(r1.manufacturer)), 
                                set(tokenize(r2.manufacturer)))

    if r1.price is None or r2.price is None:
        fv['price_difference'] = None
    else:
        fv['price_difference'] = abs(r1.price - r2.price)/max(r1.price, r2.price)

    fv['name_jaccard'] = rltk.jaccard_index_similarity(set(r1.name_tokenized), set(r2.name_tokenized))
    fv['name_jaro_winkler'] = rltk.jaro_winkler_similarity(" ".join(r1.name_tokenized), " ".join(r2.name_tokenized))
    fv['name_trigram'] = rltk.ngram_distance(r1.name, r2.name,3)
    
    if r1.description_tokenized is None or r2.description_tokenized is None:
        fv['desc_tf_idf'] = None
        fv['desc_trigram'] = None
        fv['desc_jaccard'] = None
    else:
        fv['desc_tf_idf'] = rltk.tf_idf_similarity(r1.description_tokenized,
                                                r2.description_tokenized,freq,doc_size)
        fv['desc_trigram'] = rltk.ngram_distance(" ".join(r1.description_tokenized), " ".join(r2.description_tokenized),3)
        fv['desc_jaccard'] = rltk.jaccard_index_similarity(set(r1.description_tokenized), set(r2.description_tokenized))

    return fv

def featurize_all_records(pairs, features, output_filename, freq, doc_size):
    """
    Featurize all records and save result in CSV

    Params:
        pairs: pairs obtained from rltk.get_record_pairs()
        features: (List) features to be considered
        output_filename: (str) name of output CSV file
        freq: (Dict) corpus frequency
        doc_size: (int) total size of dataset
    """
    feature_vectors = pd.DataFrame(columns=features)
    i = 0
    for r1,r2 in pairs:
        i+= 1
        if i%1000 == 0:
            print("Progress = {} records".format(i))

        fv = featurize_record_pair(r1, r2, freq, doc_size)
        feature_vectors = feature_vectors.append(fv, ignore_index=True)
        
        if i % 1000 == True:
            feature_vectors.to_csv(output_filename,index=None)
        
    feature_vectors.to_csv(output_filename,index=None)

def get_document_frequency(filename, ds_amzn, ds_goog):
    """
    Get document frequency of corpus

    Params:
        filename: (str) path of corpus frequency file
        ds_amzn: (rltk.Dataset) Amazon dataset obj
        ds_goog: (rltk.Dataset) Google dataset obj
    """
    try:
        freq = open(filename,'r')
        try:
            freq = json.load(freq)
            return freq
        except Exception as e:
            print(e)
    except FileNotFoundError:
        print("Corpus frequency file not found. Calculating ...")

    corpus = []
    for r in ds_amzn:
        if r.description_tokenized is not None and r.name_tokenized is not None:
            corpus.extend(set(r.description_tokenized + r.name_tokenized))
    
    for r in ds_goog:
        if r.description_tokenized is not None:
            corpus.extend(set(r.description_tokenized + r.name_tokenized))
    
    freq = dict(Counter(corpus))

    with open(filename, 'w') as f:
        json.dump(freq, f)

    return freq

def featurize(mode, output_filename=None):
    """
    Catch all method to featurize either train or test dataset and save to CSV

    Params:
        mode: (str) TRAIN or TEST
        output_filename: (str) Optional- name of the csv to save the data
    """
    MODE = mode
    if not os.path.exists('train/') or not os.path.exists('test/'):
        train_test_split()
        
    if not os.path.exists('block_files/'):
        os.mkdir('block_files/')

    BLOCK_FILE = 'block_files/'+MODE+'.jl'
    CORPUS_FREQ_FILE = MODE+'/corpus_freq.json'

    ds_amzn = rltk.Dataset(reader=rltk.CSVReader(open(MODE + '/Amazon.csv', encoding='latin-1')),
                    record_class=AmazonRecord, adapter=rltk.MemoryAdapter())

    ds_goog = rltk.Dataset(reader=rltk.CSVReader(open(MODE + '/GoogleProducts.csv', encoding='latin-1')),
                    record_class=GoogleRecord, adapter=rltk.MemoryAdapter())

    try:
        block_handler = open(BLOCK_FILE,'r')
        print("Block file exists. Reading from disk...")
    except FileNotFoundError:
        block_handler = rltk.InvertedIndexBlockGenerator(
            ds_amzn, ds_goog, writer=rltk.BlockFileWriter(BLOCK_FILE), tokenizer=tokenizer).generate()

    features = ['id1', 'id2', 'price_difference',
       'desc_jaccard', 'desc_tf_idf', 'desc_trigram',
       'manufacturer_jaccard', 'manufacturer_jaro_winkler',
       'manufacturer_levenshtien', 'name_jaccard', 'name_jaro_winkler',
       'name_trigram','label']

    pairs = rltk.get_record_pairs(ds_amzn, ds_goog, rltk.BlockFileReader(block_handler))
    freq = get_document_frequency(CORPUS_FREQ_FILE, ds_amzn, ds_goog)

    if MODE == "train":
        print("Featurizing train")
        if not output_filename:
            output_filename = 'train/features_train.csv'
        featurize_all_records(pairs, features, output_filename, freq, TRAIN_DOC_SIZE)
    elif MODE == "test":
        print("Featurizing test")
        if not output_filename:
            output_filename = 'test/features_test.csv'
        featurize_all_records(pairs, features, output_filename, freq, TEST_DOC_SIZE)

TRAIN_DOC_SIZE = 3444
TEST_DOC_SIZE = 1149
gt = rltk.GroundTruth()
gt.load(DATASET_DIR+"Amzon_GoogleProducts_perfectMapping.csv")

def train_test_split():

    os.mkdir('train/')
    os.mkdir('test/')

    amazon = pd.read_csv(DATASET_DIR+'Amazon.csv', encoding='latin-1')
    google = pd.read_csv(DATASET_DIR+'GoogleProducts.csv', encoding='latin-1')

    # Fraction of dataset to split on
    fraction = 0.75

    train = amazon.sample(frac=fraction)
    train.to_csv('train/Amazon.csv',index=None)
    test = amazon[~amazon.id.isin(train.id)]
    # Ensure train test sets are disjoint
    assert train.id.isin(test.id).any() == False, "Train test not disjoint!"
    test.to_csv('test/Amazon.csv',index=None)

    train = google.sample(frac=fraction)
    train.to_csv('train/GoogleProducts.csv',index=None)
    test = google[~google.id.isin(train.id)]
    assert train.id.isin(test.id).any() == False
    test.to_csv('test/GoogleProducts.csv',index=None)

if __name__ == '__main__':

    if sys.argv[1] == '--train':
        MODE = 'train'
    elif sys.argv[1] == '--test':
        MODE = 'test'

    featurize(mode=MODE)