from create_datasets import *

# def generate_feature_vector(r_abt, r_buy):
#     brand_score = 0.5
#     if r_abt.brand_cleaned and r_buy.brand_cleaned:
#         if r_abt.brand_cleaned == r_buy.brand_cleaned:
#             brand_score = 1
#         else:
#             if len(r_abt.brand_cleaned) >= len(r_buy.brand_cleaned):
#                 common_str = r_buy.brand_cleaned
#                 if r_abt.brand_cleaned.startswith(common_str) or r_abt.brand_cleaned.endswith(common_str):
#                     brand_score = 1
#                 else:
#                     brand_score = 0
#     model_score = 0.5
#     if r_abt.model_cleaned and r_buy.model_cleaned:
#         if r_abt.model_cleaned == r_buy.model_cleaned:
#             model_score = 1
#         else:
#             if len(r_abt.model_cleaned) >= len(r_buy.model_cleaned):
#                 common_str = r_buy.model_cleaned
#                 if r_abt.model_cleaned.startswith(common_str) or r_abt.model_cleaned.endswith(common_str):
#                     model_score = 1
#                 else:
#                     model_score = 0
#
#     if brand_score == 1 and model_score == 1:
#         jaccard_score = 1
#     else:
#         jaccard_score = rltk.jaccard_index_similarity(r_abt.name_tokens, r_buy.name_tokens)
#
#     return [brand_score, model_score, jaccard_score]



def generate_feature_vector_raw(r_abt, r_buy):
    # brand
    brand_score = None
    if r_abt.brand_cleaned and r_buy.brand_cleaned:
        if r_abt.brand_cleaned == r_buy.brand_cleaned:
            brand_score = 1

    # model 1
    model_score = None
    model_marker = 0
    if r_abt.model_cleaned and r_buy.model_cleaned:
        if r_abt.model_cleaned == r_buy.model_cleaned:
            model_score = 1
        else:
            if len(r_abt.model_cleaned) > len(r_buy.model_cleaned):
                if r_abt.model_cleaned.startswith(r_buy.model_cleaned) \
                        or r_abt.model_cleaned.endswith(r_buy.model_cleaned):
                    model_score = 1
                else:
                    model_score = rltk.levenshtein_similarity(r_abt.model_cleaned, r_buy.model_cleaned)
            elif len(r_abt.model_cleaned) < len(r_buy.model_cleaned):
                if r_buy.model_cleaned.startswith(r_abt.model_cleaned) \
                        or r_buy.model_cleaned.endswith(r_abt.model_cleaned):
                    model_score = 1
                else:
                    model_score = rltk.levenshtein_similarity(r_abt.model_cleaned, r_buy.model_cleaned)
            else:
                model_score = 0

    # model 2
    model2_score = rltk.levenshtein_similarity(r_abt.model_cleaned, r_buy.name_suffix)

    # name tokens jaccard
    jaccard_score = rltk.jaccard_index_similarity(r_abt.name_tokens, r_buy.name_tokens)

    # name tokens tf-idf
    # t_x = collections.Counter(r_abt.name_tokens)
    # tf_x = {k: float(v) / len(r_abt.name_tokens) for k, v in t_x.items()}
    # tfidf_x = {k : tf_x[k] / idf[k] for k, v in tf_x.items()}
    # t_y = collections.Counter(r_buy.name_tokens)
    # tf_y = {k: float(v) / len(r_buy.name_tokens) for k, v in t_y.items()}
    # tfidf_y = {k : tf_y[k] / idf[k] for k, v in tf_y.items()}
    # tfidf_score = rltk.tf_idf_similarity_by_dict(tfidf_x, tfidf_y)
    tfidf_score = tfidf.similarity(r_abt.id, r_buy.id)

    # price
    if r_abt.price and r_buy.price:
        price_marker = 1
        abt_price = float(r_abt.price)
        buy_price = float(r_buy.price)
        if abt_price == 0 and buy_price == 0:
            price_difference = 0
        else:
            price_difference = float(abs(abt_price - buy_price)) / max(abt_price, buy_price)
    else:
        price_marker = 0
        price_difference = 0

    return [brand_score, model_score,
            model2_score, jaccard_score, tfidf_score, price_difference, price_marker]


def generate_feature_vector(r_abt, r_buy):
    # brand
    brand_score = 0.2
    brand_marker = 0
    if r_abt.brand_cleaned and r_buy.brand_cleaned:
        if r_abt.brand_cleaned == r_buy.brand_cleaned:
            brand_score = 1
            brand_marker = 1


    # model 1
    model_score = 0.2
    model_marker = 0
    if r_abt.model_cleaned and r_buy.model_cleaned:
        if r_abt.model_cleaned == r_buy.model_cleaned:
            model_score = 1
            model_marker = 1
        else:
            if len(r_abt.model_cleaned) > len(r_buy.model_cleaned):
                if r_abt.model_cleaned.startswith(r_buy.model_cleaned) \
                        or r_abt.model_cleaned.endswith(r_buy.model_cleaned):
                    model_score = 1
                    model_marker = 1
                else:
                    model_score = rltk.levenshtein_similarity(r_abt.model_cleaned, r_buy.model_cleaned)
            elif len(r_abt.model_cleaned) < len(r_buy.model_cleaned):
                if r_buy.model_cleaned.startswith(r_abt.model_cleaned) \
                        or r_buy.model_cleaned.endswith(r_abt.model_cleaned):
                    model_score = 1
                    model_marker = 1
                else:
                    model_score = rltk.levenshtein_similarity(r_abt.model_cleaned, r_buy.model_cleaned)
            else:
                model_score = 0

    # model 2
    model2_score = rltk.levenshtein_similarity(r_abt.model_cleaned, r_buy.name_suffix)

    # name tokens jaccard
    jaccard_score = rltk.jaccard_index_similarity(r_abt.name_tokens, r_buy.name_tokens)

    tfidf_score = tfidf.similarity(r_abt.id, r_buy.id)

    # price
    if r_abt.price and r_buy.price:
        price_marker = 1
        abt_price = float(r_abt.price)
        buy_price = float(r_buy.price)
        if abt_price == 0 and buy_price == 0:
            price_difference = 0
        else:
            price_difference = float(abs(abt_price - buy_price)) / max(abt_price, buy_price)
    else:
        price_marker = 0
        price_difference = 0

    return [brand_score, brand_marker, model_score, model_marker,
            model2_score, jaccard_score, tfidf_score, price_difference, price_marker]


def non_ml_method(r_abt, r_buy):
    brand_score = 0
    if r_abt.brand_cleaned and r_buy.brand_cleaned:
        if r_abt.brand_cleaned == r_buy.brand_cleaned:
            brand_score = 1
    model_score = 0
    if r_abt.model_cleaned and r_buy.model_cleaned:
        if r_abt.model_cleaned == r_buy.model_cleaned:
            model_score = 1
    jaccard_score = rltk.jaccard_index_similarity(r_abt.name_tokens, r_buy.name_tokens)

    if model_score == 1:
        return True

    total = brand_score * 0.3 + model_score * 0.3 + jaccard_score * 0.4
    return total > 0.45