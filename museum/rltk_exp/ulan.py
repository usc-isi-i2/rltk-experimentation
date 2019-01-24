import re
import time
import rltk


name_filter = re.compile('[^A-Za-z0-9 ]+')


def tokenize_name(name):
    name = name.strip().lower()

    # Keep only alpha numerics
    name = name_filter.sub('', name)

    # extract all space separated tokens from the names
    return set([w for w in name.split(' ')])


@rltk.remove_raw_object
class RecordAAA(rltk.Record):
    @rltk.cached_property
    def id(self):
        return self.raw_object['uri']['value']

    @rltk.cached_property
    def name(self):
        return self.raw_object['name']['value']

    @rltk.cached_property
    def name_tokens(self):
        return tokenize_name(self.raw_object['name']['value'])

    @rltk.cached_property
    def birthday(self):
        if 'byear' in self.raw_object:
            return self.raw_object['byear']['value']
        return None

    @rltk.cached_property
    def birthyear(self):
        if 'byear' in self.raw_object:
            return self.raw_object['byear']['value'][:4]
        return None


@rltk.remove_raw_object
class RecordULAN(rltk.Record):
    @rltk.cached_property
    def id(self):
        return self.raw_object['uri']['value']

    @rltk.cached_property
    def name(self):
        return self.raw_object['name']['value']

    @rltk.cached_property
    def name_tokens(self):
        return tokenize_name(self.raw_object['name']['value'])

    @rltk.cached_property
    def birthyear(self):
        return self.raw_object['byear']['value']


ds_aaa = rltk.Dataset(reader=rltk.JsonLinesReader('../../datasets/museum/aaa.json'), record_class=RecordAAA, size=1000)
ds_ulan = rltk.Dataset(reader=rltk.JsonLinesReader('../../datasets/museum/ulan.json'), record_class=RecordULAN, size=1000)


def block_on_name_prefix(r):
    return [n[:2] for n in r.name_tokens]


bg = rltk.TokenBlockGenerator()
b_aaa = bg.block(ds_aaa, function_=block_on_name_prefix)
b_ulan = bg.block(ds_ulan, function_=block_on_name_prefix)
b_aaa_ulan = bg.generate(b_aaa, b_ulan)


def compare(r_aaa, r_ulan):
    # if birth year exists and not equal, exact not match
    if r_aaa.birthyear and r_ulan.birthyear:
        if r_aaa.birthyear != r_ulan.birthyear:
            return 0

    return rltk.hybrid_jaccard_similarity(r_aaa.name_tokens, r_ulan.name_tokens, threshold=0.3)


def output_handler(*arg):
    if arg[0]:
        r_aaa, r_ulan = arg[1], arg[2]
        print(r_aaa.name, r_ulan.name)


# time_start = time.time()
# pp = rltk.ParallelProcessor(is_pair, 8)
# pp.start()
#
# for idx, (r_aaa, r_ulan) in enumerate(rltk.get_record_pairs(ds_aaa, ds_ulan)):
#     print(idx)
#     pp.compute(r_aaa, r_ulan)
#
# pp.task_done()
# pp.join()
# time_pp = time.time() - time_start
# print('pp time:', time_pp)

match = {}
threshold = 0.67
time_start = time.time()
for idx, (r_aaa, r_ulan) in enumerate(rltk.get_record_pairs(ds_aaa, ds_ulan, block=b_aaa_ulan)):
    if idx % 10000 == 0:
        print(idx)

    score = compare(r_aaa, r_ulan)
    if score > threshold:
        prev = match.get(r_aaa.id, [0, 'dummy ulan id'])
        if score > prev[0]:
            match[r_aaa.id] = [score, r_ulan.id]

time_normal = time.time() - time_start
print('normal time:', time_normal)
print(len(match))
print(match)
