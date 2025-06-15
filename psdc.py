#!/usr/bin/env python3
# coding: utf-8

import re
import nltk
import math
import logging
import numpy as np
import polars as pl
import fasttext.util

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def _clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))


def _cosine_similarity(a, b):
    return np.dot(a,b) / np.linalg.norm(a) * np.linalg.norm(b)


def _cosine_distance(a, b):
    return 1.0 - _clamp(_cosine_similarity(a, b), -1.0, 1.0)


def _nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif nltk_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:          
        return None


def _nltk_pos_lemmatizer(token, tag=None):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    if tag is None:
        return lemmatizer.lemmatize(token)
    else:        
        return lemmatizer.lemmatize(token, tag)


def _text_pre_processing(txt, m=1):
    if txt is not None:
        txt = re.sub('[-_.]', ' ', txt)
        txt = re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', txt)

        tokens = nltk.word_tokenize(txt)
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if w.isalpha()]
        tokens = [w for w in tokens if len(w) > m]
        tokens = nltk.pos_tag(tokens)
        tokens = [(t[0], _nltk_pos_tagger(t[1])) for t in tokens]
        tokens = [_nltk_pos_lemmatizer(w, t) for w,t in tokens]
    else:
        tokens = []
    
    return tokens


class NumericalFeature:
    def __init__(self, min_value, max_value, hist):
        self.min = min_value
        self.max = max_value
        self.hist = hist
    
    def distance(self, nf, weights=None):
        if weights is None:
            weights = [1/3, 1/3, 1/3]
        return (weights[0] * math.fabs(self.min - nf.min) + weights[1] * math.fabs(self.max - nf.max) + weights[2] * _cosine_distance(self.hist, nf.hist))/np.sum(weights)


class SemanticFeature:
    def __init__(self, vector):
        self.vector = vector

    def distance(self, sf):
        return _cosine_distance(self.vector, sf.vector)


class Feature:
    def __init__(self, sf, nf, cat, dn, fn):
        self.sf = sf
        self.nf = nf
        self.cat = cat
        self.dn = dn
        self.fn = fn

    def distance(self, ft, weights=None, bias=None, weights_num=None):
        if weights is None:
            weights = [1/2, 1/2]
        
        if bias is None:
            bias = [0.0, 0.0]
        
        if weights_num is None:
            weights_num = [1/3, 1/3, 1/3]

        if self.nf is None or ft.nf is None:
            #TODO: Categorical distance
            num_part = weights[1] * bias[1]
        else:
            num_part = weights[1] * self.nf.distance(ft.nf, weights=weights_num)
        
        sem_part = weights[0] * self.sf.distance(ft.sf)

        logger.debug(f'{sem_part} {num_part}')

        return (sem_part + num_part) / np.sum(weights)


class PsDCModel:
    def __init__(self, weights=None, weights_num=None, bias=None, datasets_path='datasets', lm=None):
        self.datasets_path=datasets_path
        if lm is None:
            fasttext.util.download_model('en', if_exists='ignore')  # English
            self.lm = fasttext.load_model('cc.en.300.bin')
        else:
            self.lm = lm
        self.samples = []
        if weights is None:
            weights = [1/2, 1/2]
        self.weights = weights

        if weights_num is None:
            weights_num = [1/3, 1/3, 1/3]
        self.weights_num = weights_num

        if bias is None:
            bias = [0.0, 0.0]
        self.bias = bias
    
    def fit(self, metadata):
        for dataset_name in metadata:
            logger.debug(f'{dataset_name}')
            dataset = metadata[dataset_name]
            # load dataset with polars
            df = pl.read_csv(f'{self.datasets_path}/{dataset_name}.csv.gz', separator=',', has_header=True)
            logger.debug(f'{df.head}')
            for feature_name in dataset:
                arr = df.get_column(feature_name).to_numpy()
                feature = dataset[feature_name]
                
                # text
                tokens = _text_pre_processing(feature_name)
                sentence = ' '.join(tokens)
                logger.debug(f'\t{feature_name} -> {tokens} {sentence}')
                vector = self.lm.get_sentence_vector(sentence)
                sf = SemanticFeature(vector)

                # numerical
                if arr.dtype.kind in 'iuf':
                    # prepare data
                    arr = np.where(np.isnan(arr), np.nanmedian(arr), arr)
                    h, _ = np.histogram(arr, bins=10)
                    h = h/h.sum()
                    #logger.info(f'\t{feature_name}: numerical [{np.min(arr)} - {np.max(arr)}] {h}')
                    nf = NumericalFeature(np.min(arr), np.max(arr), h)
                else:
                    # categorical
                    nf = None
                cat = feature
                self.samples.append(Feature(sf, nf, cat, dataset_name, feature_name))
    
    def predict(self, ft):
        min_dist = ft.distance(self.samples[0])
        min_idx = 0
        for  i in range(1, len(self.samples)):
            current_distance = ft.distance(self.samples[i], weights=self.weights, bias=self.bias, weights_num=self.weights_num)
            if current_distance < min_dist:
                min_dist = current_distance
                min_idx = i
        return self.samples[min_idx].cat, min_dist, f'{self.samples[min_idx].dn}/{self.samples[min_idx].fn}'

    def most_frequent_categories(self):
        mfc = {}
        for sample in self.samples:
            cats = sample.cat
            for cat in cats:
                if cat not in mfc:
                    mfc[cat] = 0
                mfc[cat] += 1
        return mfc

    def get_features(self, mfc):
        features = []
        for cat, _ in mfc:
            for i in range(0, len(self.samples)):
                sample = self.samples[i]
                if cat in sample.cat:
                    features.append(sample)
                    self.samples.pop(i)
                    break
        return features

    def get_accuracy(self, test_features):
        total = 0.0
        acc = 0.0
        for f in test_features:
            total += 1.0
            cat, dist, debug = self.predict(f)
            cats = list(cat.keys())
            for c in cats:
                if c in f.cat:
                    acc += 1.0
                    break
        return acc/total