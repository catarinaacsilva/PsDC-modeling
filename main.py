#!/usr/bin/env python3
# coding: utf-8


import json
import psdc
import optuna
import logging


optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def psdc_objective(trial, metadata):
    wsem = trial.suggest_float('wsem', 0.0, 1.0)
    wnum1 = trial.suggest_float('wnum1', 0.0, 1.0)
    wnum2 = trial.suggest_float('wnum2', 0.0, 1.0-wnum1)
    bsem = trial.suggest_float('bsem', 0.0, 1.0)
    
    weights=[wsem, 1.0-wsem]
    weights_num=[wnum1, wnum2, 1.0-(wnum1+wnum2)]
    bias = [bsem, 1.0-bsem]

    model = psdc.PsDCModel(weights=weights, weights_num=weights_num, bias=bias)
    model.fit(metadata)

    mfc = model.most_frequent_categories()
    mfc = list(mfc.items())
    mfc = sorted(mfc, key=lambda tup: tup[1], reverse=True)
    mfc = mfc[:50]
    test_features = model.get_features(mfc)
    acc = model.get_accuracy(test_features)

    logger.info(f'{weights} {weights_num} {bias} -> {acc}')

    return acc


def train_psdc(metadata):
    # Running the Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: psdc_objective(trial, metadata), n_trials=100)

    # Get the best model and hyperparameters
    best_params_optuna = study.best_params

    logger.info(best_params_optuna)
    return best_params_optuna


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
    
    # Hyperparameter tunning
    best_params_optuna = train_psdc(metadata)
    #best_params_optuna = {'wsem': 0.4, 'wnum1': 0.8, 'wnum2': 0.01, 'bsem': 0.8} # Optimal
    #best_params_optuna = {'wsem': 1/2, 'wnum1': 1/3, 'wnum2': 1/3, 'bsem': 1}    # Normal
    #best_params_optuna = {'wsem': 1, 'wnum1': 1/3, 'wnum2': 1/3, 'bsem': 1}      # Semantic
    #best_params_optuna = {'wsem': 0, 'wnum1': 1/3, 'wnum2': 1/3, 'bsem': 1}      # Numerical
    weights=[best_params_optuna['wsem'], 1.0-best_params_optuna['wsem']]
    weights_num=[best_params_optuna['wnum1'],best_params_optuna['wnum2'],1.0-(best_params_optuna['wnum1']+best_params_optuna['wnum2'])]
    bias=[best_params_optuna['bsem'], 1.0-best_params_optuna['bsem']]

    model = psdc.PsDCModel(weights=weights, weights_num=weights_num, bias=bias)

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
