#!/usr/bin/env python3
# coding: utf-8


import json
import psdc
import optuna
import logging
import numpy as np
import fasttext.util
import matplotlib.pyplot as plt


from textwrap import wrap


optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def psdc_objective(trial, metadata, lm):
    wsem = trial.suggest_float('wsem', 0.0, 1.0)
    wnum1 = trial.suggest_float('wnum1', 0.0, 1.0)
    wnum2 = trial.suggest_float('wnum2', 0.0, 1.0-wnum1)
    bsem = trial.suggest_float('bsem', 0.0, 1.0)
    bnum = trial.suggest_float('bnum', 0.0, 1.0)
    
    weights=[wsem, 1.0-wsem]
    weights_num=[wnum1, wnum2, 1.0-(wnum1+wnum2)]
    bias = [bsem, bnum]

    model = psdc.PsDCModel(weights=weights, weights_num=weights_num, bias=bias, lm=lm)
    model.fit(metadata)

    mfc = model.most_frequent_categories()
    mfc = list(mfc.items())
    mfc = sorted(mfc, key=lambda tup: tup[1], reverse=True)
    mfc = mfc[:32]
    test_features = model.get_features(mfc)
    acc = model.get_accuracy(test_features)

    logger.debug(f'{weights} {weights_num} {bias} -> {acc}')

    return acc


def train_psdc(metadata, lm):
    # Running the Optuna optimization
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(lambda trial: psdc_objective(trial, metadata, lm), n_trials=200, show_progress_bar=True)

    # Get the best model and hyperparameters
    best_params_optuna = study.best_params

    logger.info(best_params_optuna)
    return best_params_optuna


def most_frequent_categories(metadata):
    mfc = {}

    for dataset_name in metadata:
        dataset = metadata[dataset_name]
        for field_name in dataset:
            field = dataset[field_name]
            for category_name in field:
                if category_name != 'NA':
                    if category_name not in mfc:
                        mfc[category_name] = 0.0
                    mfc[category_name] += 1
    
    total = 0.0
    for category_name in mfc:
        total += mfc[category_name]
    
    for category_name in mfc:
        mfc[category_name] /= total

    mfc = list(mfc.items())
    mfc = sorted(mfc, key=lambda tup: tup[1], reverse=False)
    
    return mfc


def main():
    # load metadata
    with open('metadata.json', 'r') as file:
        metadata = json.load(file)
    logger.debug(f'{metadata}')

    # print stats
    logger.info(f'Number of datasets: {len(metadata)}')
    num_features = 0
    num_categories = 0
    sum_categories = 0
    for dataset_name in metadata:
        dataset = metadata[dataset_name]
        for feature_name in dataset:
            num_features += 1
            feature = dataset[feature_name]
            for category_name in feature:
                num_categories += 1
                sum_categories += feature[category_name]
    
    logger.info(f'Number of features: {num_features}')
    logger.info(f'Average agreement level: {sum_categories/num_categories}')

    # Compute the MFC in metadata
    mfc = most_frequent_categories(metadata)
    logger.debug(f'{mfc}')
    
    # Plot data
    x, y = zip(*mfc)

    labels = [ '\n'.join(wrap(l.replace('/', ' '), 32)).replace(' ','/').replace('\n', '/\n') for l in x ]
    logger.debug(f'{labels}')

    fig = plt.figure(layout='constrained', figsize=(12, 10)) 
    plt.barh(labels,y)
    plt.savefig('figures/categories_distribution.pdf')

    
    # Pre-load the LM
    fasttext.util.download_model('en', if_exists='ignore')  # English
    lm = fasttext.load_model('cc.en.300.bin')

    # Hyperparameter tunning
    best_params_optuna = train_psdc(metadata, lm)
    #best_params_optuna = {'wsem': 0.5, 'wnum1': 0.9, 'wnum2': 0.01, 'bsem': 0.3, 'bnum': 0.2} # Optimal
    #best_params_optuna = {'wsem': 1/2, 'wnum1': 1/3, 'wnum2': 1/3, 'bsem': 1.0, 'bnum': 1.0}  # Normal
    #best_params_optuna = {'wsem': 1.0, 'wnum1': 1/3, 'wnum2': 1/3, 'bsem': 1.0, 'bnum': 1.0}  # Semantic
    #best_params_optuna = {'wsem': 0.0, 'wnum1': 1/3, 'wnum2': 1/3, 'bsem': 1.0, 'bnum': 1.0}  # Numerical
    weights=[best_params_optuna['wsem'], 1.0-best_params_optuna['wsem']]
    weights_num=[best_params_optuna['wnum1'],best_params_optuna['wnum2'],1.0-(best_params_optuna['wnum1']+best_params_optuna['wnum2'])]
    bias=[best_params_optuna['bsem'], best_params_optuna['bnum']]

    model = psdc.PsDCModel(weights=weights, weights_num=weights_num, bias=bias, lm=lm)

    # fit model
    model.fit(metadata)

    # get test set (13 samples, arround 10%)
    mfc = model.most_frequent_categories()
    mfc = list(mfc.items())
    mfc = sorted(mfc, key=lambda tup: tup[1], reverse=True)
    mfc = mfc[:13]
    #logger.info(f'{mfc}')

    test_features = model.get_features(mfc)
    #logger.info(f'{test_features}')

    acc = model.get_accuracy(test_features)
    logger.info(f'accuracy {acc}')


if __name__ == '__main__':
    main()
