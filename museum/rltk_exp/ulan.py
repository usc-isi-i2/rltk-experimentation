import re
import sys
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
class RecordMuseum(rltk.Record):
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

    @property
    def birthyear(self):
        return self.raw_object['byear']['value']


def block_on_name_prefix(r):
    ret = []
    for n in r.name_tokens:
        if len(n) > 2:
            ret.append(n[:2])
    return ret


def compare(r_aaa, r_ulan):
    # if birth year exists and not equal, exact not match
    if r_aaa.birthyear and r_ulan.birthyear:
        if r_aaa.birthyear != r_ulan.birthyear:
            return 0

    return rltk.hybrid_jaccard_similarity(r_aaa.name_tokens, r_ulan.name_tokens, threshold=0.67)


def output_handler(*arg):
    if arg[0]:
        r_aaa, r_ulan = arg[1], arg[2]
        print(r_aaa.name, r_ulan.name)


if __name__ == '__main__':
    INIT_ULAN = False
    ulan_ds_adapter = rltk.RedisKeyValueAdapter('127.0.0.1', key_prefix='ulan_ds_')
    bg = rltk.TokenBlockGenerator()
    ulan_block = rltk.Block(rltk.RedisKeySetAdapter('127.0.0.1', key_prefix='ulan_block_'))

    # pre computing for ulan data
    if INIT_ULAN:
        ds_ulan = rltk.Dataset(reader=rltk.JsonLinesReader('../../datasets/museum/ulan.json'),
                               record_class=RecordULAN,
                               adapter=ulan_ds_adapter)
        b_ulan = bg.block(ds_ulan, function_=block_on_name_prefix, block=ulan_block)
        exit()

    # load ulan
    ds_ulan = rltk.Dataset(adapter=ulan_ds_adapter)
    b_ulan = ulan_block

    # compare against museums' data
    museums = ['autry']
    for museum in museums:
        print('-------------------')
        print('For museum: {}'.format(museum))

        # load museum
        ds_museum = rltk.Dataset(reader=rltk.JsonLinesReader('../../datasets/museum/{}.json'.format(museum)),
                              record_class=RecordMuseum)
        b_museum = bg.block(ds_museum , function_=block_on_name_prefix)
        b_museum_ulan = bg.generate(b_museum, b_ulan)

        # statistics
        pairwise_len = sum(1 for _ in b_museum_ulan.pairwise(ds_museum.id, ds_ulan.id))
        ulan_len = sum(1 for _ in ds_ulan)
        museum_len = sum(1 for _ in ds_museum)
        print('pairwise comparison:', pairwise_len, 'ratio:', format(pairwise_len / (ulan_len * museum_len)))

        # dup = {}
        # for _, aid, uid in b_museum_ulan.pairwise(ds_museum.id, ds_ulan.id):
        #     k = '{}|{}'.format(aid, uid)
        #     if k not in dup:
        #         dup[k] = 0
        #     dup[k] += 1
        # import operator, functools
        # total = functools.reduce(operator.add, dup.values())
        # print('duplication ratio:', total / len(dup))

        # start
        print('start pairwise comparison...')
        match = {}
        threshold = 0.67

        # serial
        # time_start = time.time()
        # for idx, (r_museum, r_ulan) in enumerate(rltk.get_record_pairs(ds_museum, ds_ulan, block=b_museum_ulan)):
        #     if idx % 10000 == 0:
        #         print('\r', idx, end='')
        #         sys.stdout.flush()
        #
        #     score = compare(r_museum, r_ulan)
        #     if score > threshold:
        #         prev = match.get(r_museum.id, [0, 'dummy ulan id'])
        #         if score > prev[0]:
        #             match[r_museum.id] = [score, r_ulan.id]
        # time_normal = time.time() - time_start
        #
        # print('\r', end='')
        # print('normal time:', time_normal / 60)
        # print(len(match))
        # print(match)

        # parallel
        def mapper(r_museum, r_ulan):
            score = compare(r_museum, r_ulan)
            r_museum_id = r_museum.id
            r_ulan_id = r_ulan.id
            if score > threshold:
                return {r_museum_id: (score, r_ulan_id)}
            return {}

        def reducer(r1, r2):
            for k, v in r1.items():
                if k not in r2 or v[0] > r2[k][0]:
                    r2[k] = v
            return r2

        mr = rltk.MapReduce(8, mapper, reducer)

        time_start = time.time()
        for idx, (r_museum, r_ulan) in enumerate(rltk.get_record_pairs(ds_museum, ds_ulan, block=b_museum_ulan)):
            if idx % 10000 == 0:
                break
                print('\r', idx, end='')
                sys.stdout.flush()
            mr.add_task(r_museum, r_ulan)

        print('')
        result = mr.join()
        time_pp = time.time() - time_start
        print('pp time:', time_pp / 60)
