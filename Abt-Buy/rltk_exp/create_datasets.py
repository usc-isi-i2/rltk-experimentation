import rltk

tokenizer = rltk.CrfTokenizer()


model_stop_words = set([])
with open('stop_words_model.txt') as f:
    for line in f:
        line = line.strip().lower()
        if line:
            model_stop_words.add(line)


def extract_possible_model(s):
    possible_models = []
    tokens = s.split(' ')

    for t in tokens:
        t = t.replace('(', '').replace(')', '')
        if len(t) < 2 or t in model_stop_words:
            continue

        if t.isdigit():
            possible_models.append(t)
            continue

        has_digit = has_alpha = False
        for c in t:
            if c.isdigit():
                has_digit = True
            elif c.isalpha():
                has_alpha = True
            if has_digit and has_alpha:
                possible_models.append(t)

    possible_models.sort(key=len, reverse=True)

    return possible_models[0] if len(possible_models) > 0 else ''


def tokenize(s):
    tokens = tokenizer.tokenize(s)
    return [w.lower() for w in tokens if w.isalpha()]


def get_brand_name(tokens):
    for word_len in range(min(5, len(tokens)), 0, -1):
        i = 0; j = i + word_len
        while j <= len(tokens):
            name = ' '.join(tokens[i:j])
            if name in brand_list:
                return name
            i += 1; j += 1
    return ''


def process_brand_alias(alias):
    return brand_mapping.get(alias, alias)


brand_list = set([])
with open('brands.txt') as f:
    for line in f:
        line = line.strip().lower()
        if len(line) == 0:
            continue
        brand_list.add(' '.join(tokenize(line)))

brand_mapping = {}
with open('brand_alias.txt') as f:
    for line in f:
        alias = [w.strip().lower() for w in line.split('|')]
        for name in alias:
            brand_mapping[name] = alias[0]


class AbtRecord(rltk.Record):
    def __init__(self, raw_object):
        super().__init__(raw_object)
        self.brand = ''

    @rltk.cached_property
    def id(self):
        return self.raw_object['id']

    @rltk.cached_property
    def name(self):
        return self.raw_object['name'].split(' - ')[0]

    @rltk.cached_property
    def name_tokens(self):
        tokens = tokenize(self.name)

        self.brand = get_brand_name(tokens)

        return set(tokens)

    @rltk.cached_property
    def model(self):
        ss = self.raw_object['name'].split(' - ')
        return ss[-1].strip() if len(ss) > 1 else ''

    @rltk.cached_property
    def description(self):
        return self.raw_object.get('description', '')

    @rltk.cached_property
    def price(self):
        p = self.raw_object.get('price', '')
        if p.startswith('$'):
            p = p[1:].replace(',', '')
        return p

    @rltk.cached_property
    def brand_cleaned(self):
        _ = self.name_tokens
        return process_brand_alias(self.brand)

    @rltk.cached_property
    def model_cleaned(self):
        m = self.model
        return m.lower().replace('-', '').replace('/', '').replace('&', '')


class BuyRecord(rltk.Record):
    def __init__(self, raw_object):
        super().__init__(raw_object)
        self.brand = ''

    @rltk.cached_property
    def id(self):
        return self.raw_object['id']

    @rltk.cached_property
    def name(self):
        return self.raw_object['name'].split(' - ')[0]

    @rltk.cached_property
    def name_tokens(self):
        tokens = tokenize(self.name)
        self.brand = get_brand_name(tokens)
        return set(tokens)

    @rltk.cached_property
    def description(self):
        return self.raw_object.get('description', '')

    @rltk.cached_property
    def manufacturer(self):
        return self.raw_object.get('manufacturer', '').lower()

    @rltk.cached_property
    def price(self):
        p = self.raw_object.get('price', '')
        if p.startswith('$'):
            p = p[1:].replace(',', '')
        return p

    @rltk.cached_property
    def model(self):
        ss = self.raw_object['name'].split(' - ')
        ss = ss[0].strip()

        return extract_possible_model(ss)

    @rltk.cached_property
    def name_suffix(self): # could probably be the model
        ss = self.raw_object['name'].split(' - ')
        return BuyRecord._clean(ss[-1]) if len(ss) > 1 else ''

    @staticmethod
    def _clean(s):
        return s.lower().replace('-', '').replace('/', '').replace('&', '')

    @rltk.cached_property
    def brand_cleaned(self):
        _ = self.name_tokens
        manufacturer = self.manufacturer
        return process_brand_alias(manufacturer if manufacturer != '' else self.brand)

    @rltk.cached_property
    def model_cleaned(self):
        m = self.model
        return BuyRecord._clean(m)


ds_abt = rltk.Dataset(reader=rltk.CSVReader(open('../../datasets/Abt-Buy/Abt.csv', encoding='latin-1')),
                   record_class=AbtRecord, adapter=rltk.MemoryAdapter())

ds_buy = rltk.Dataset(reader=rltk.CSVReader(open('../../datasets/Abt-Buy/Buy.csv', encoding='latin-1')),
                   record_class=BuyRecord, adapter=rltk.MemoryAdapter())

# statistics
print_details = False
name_count = model_count = description_count = price_count = brand_count = 0
for r in ds_abt:
    name_count += 1
    print('------\nname:', r.name) if print_details else ''
    if len(r.description) > 0:
        description_count += 1
    if len(r.price) > 0:
        price_count += 1
    if len(r.model) > 0:
        model_count += 1
        print('model:', r.model)  if print_details else ''
    if len(r.brand) > 0:
        brand_count += 1
        print('brand:', r.brand)  if print_details else ''
    else:
        print('no brand') if print_details else ''
name_count = float(name_count)
print('description:', description_count / name_count,
      'price:', price_count / name_count,
      'brand', brand_count / name_count,
      'model', model_count / name_count)

# cat abt_buy_perfectMapping.csv |  awk '{split($0,a,"," ); print a[2]}' | sort | uniq -c | grep "2 "


name_count = description_count = price_count = brand_count = model_count = manufacturer_count = 0
for r in ds_buy:
    name_count += 1
    print('------\nname:', r.name) if print_details else ''
    if len(r.description) > 0:
        description_count += 1
    if len(r.price) > 0:
        price_count += 1
    if len(r.model) > 0:
        model_count += 1
        print('model:', r.model) if print_details else ''
    if len(r.brand) > 0:
        brand_count += 1
        print('brand:', r.brand) if print_details else ''
    else:
        print('no brand') if print_details else ''
    if len(r.manufacturer) > 0:
        manufacturer_count += 1
        # print('manufacturer:', r.manufacturer)
    # else:
    #     print('no manufacturer:', r.name)

name_count = float(name_count)
print('description:', description_count / name_count,
      'price:', price_count / name_count,
      'brand', brand_count / name_count,
      'model', model_count / name_count,
      'manufacturer', manufacturer_count / name_count)


# doc_size = 0
# corpus = {}
# for r in ds_abt:
#     for t in r.name_tokens:
#         corpus[t] = corpus.get(t, 0) + 1
#     doc_size += 1
# for r in ds_buy:
#     for t in r.name_tokens:
#         corpus[t] = corpus.get(t, 0) + 1
#     doc_size += 1
# idf = rltk.compute_idf(corpus, doc_size)

tfidf = rltk.TF_IDF()
for r in ds_abt:
    tfidf.add_document(r.id, r.name_tokens)
for r in ds_buy:
    tfidf.add_document(r.id, r.name_tokens)
tfidf.pre_compute()
