#!/usr/bin/env python3
# coding: utf-8


import json
import psdc
import logging
import cProfile, pstats
import fasttext.util


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def psdc_objective(params, metadata, lm): 
    weights=[params['wsem'], 1.0-params['wsem']]
    weights_num=[params['wnum1'], params['wnum2'], 1.0-(params['wnum1'] + params['wnum2'])]
    bias = [params['bsem'], params['bnum']]

    model = psdc.PsDCModel(weights=weights, weights_num=weights_num, bias=bias, lm=lm)
    model.fit(metadata)

    mfc = model.most_frequent_categories()
    mfc = list(mfc.items())
    mfc = sorted(mfc, key=lambda tup: tup[1], reverse=True)
    mfc = mfc[:26]
    test_features = model.get_features(mfc)
    acc = model.get_accuracy(test_features)

    logger.debug(f'{weights} {weights_num} {bias} -> {acc}')

    return acc


def main():
    # load metadata
    with open('metadata.json', 'r') as file:
        metadata = json.load(file)
    logger.debug(f'{metadata}')

    params = {'wsem': 0.4, 'wnum1': 0.8, 'wnum2': 0.01, 'bsem': 0.8, 'bnum': 0.2}

    # Pre-load the LM
    fasttext.util.download_model('en', if_exists='ignore')  # English
    lm = fasttext.load_model('cc.en.300.bin')

    with cProfile.Profile() as pr:
        psdc_objective(params, metadata, lm)
        pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(30)


if __name__ == '__main__':
    main()