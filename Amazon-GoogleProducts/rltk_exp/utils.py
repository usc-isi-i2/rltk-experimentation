import rltk

DATASET_DIR = '../../datasets/Amazon-GoogleProducts/'

tokenizer = rltk.CrfTokenizer()

stopwords = set()
with open('stopwords.txt','r') as f:
    for line in f:
        stopwords.add(line.strip(' \n'))

manufacturers_list = set([])
with open('manufacturer_list.txt') as f:
    for line in f:
        line = line.strip().lower()
        if len(line) == 0:
            continue
        manufacturers_list.add(line)

aliases = dict()
with open('aliases.txt') as f:
    for line in f:
        line = line.strip().lower()
        line = line.split(' | ')
        if len(line) == 0:
            continue
        aliases[line[0]] = line[1]

def tokenize(s):
    """
    Tokenize string into words and remove words in stopwords list.
    """
    tokens = []
    for w in tokenizer.tokenize(s):
        if w not in stopwords:
            # Accept only alphabetical words greater than 2 characters
            if w.isalpha() and len(w) >= 2:
                tokens.append(w)
            # Accept 4 digit numbers (years)
            elif w.isdigit() and len(w) == 4:
                tokens.append(w)
            # Accept all words greater than 2 characters
            elif len(w)>2:
                tokens.append(w)
    return tokens

def impute_df(df):
    mask = df.manufacturer_jaro_winkler.isnull()
    df.loc[mask,'manufacturer_jaro_winkler'] = df[['name_jaro_winkler','desc_tf_idf']][mask].mean(axis=1)
    df.loc[mask,'manufacturer_levenshtien'] = df[['name_jaro_winkler','desc_tf_idf']][mask].mean(axis=1)
    df.loc[mask,'manufacturer_jaccard'] = df[['name_jaccard','desc_jaccard']][mask].mean(axis=1)
    df = df.fillna(0)
    return df

def impute_s(s):
    if not s.manufacturer_jaro_winkler:
        s.manufacturer_jaro_winkler = (s.name_jaro_winkler + s.desc_tf_idf)/2
    if not s.manufacturer_levenshtien:
        s.manufacturer_levenshtien = (s.name_jaro_winkler + s.desc_tf_idf)/2
    if not s.manufacturer_jaccard:
        s.manufacturer_jaccard = (s.name_jaccard + s.desc_jaccard)/2
    s = s.fillna(0)
    return s