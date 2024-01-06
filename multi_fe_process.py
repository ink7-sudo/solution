#!/usr/bin/env python
# coding: utf-8

import sklearn
import os
import gc
import numpy as np
import pandas as pd
import scipy
import scipy.sparse



DATA_DIR = "/home/djj/bio/solution/data/open-problems-multimodal/open-problems-multimodal"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")


class tfidfTransformer():
    def __init__(self):
        self.idf = None
        self.fitted = False

    def fit(self, X):
        self.idf = X.shape[0] / X.sum(axis=0)
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError('Transformer was not fitted on any data')
        if scipy.sparse.issparse(X):
            tf = X.multiply(1 / X.sum(axis=1))
            return tf.multiply(self.idf)
        else:
            tf = X / X.sum(axis=1).reshape(-1,1)
            return tf * self.idf

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)




X = scipy.sparse.load_npz("/home/djj/bio/solution/data/multimodal-single-cell-as-sparse-matrix/train_multi_inputs_values.sparse.npz")
Xt = scipy.sparse.load_npz("/home/djj/bio/solution/data/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_values.sparse.npz")
both = scipy.sparse.vstack([X, Xt])

from sklearn.decomposition import TruncatedSVD

pca = TruncatedSVD(n_components=512, random_state=64)
both = pca.fit_transform(both)
both -= both.mean(axis=1).reshape(-1,1)
both /= both.std(axis=1, ddof=1).reshape(-1,1)
both = both[:,:64]
X = both[:105942]
Xt = both[105942:]
del both
gc.collect()


pd.DataFrame(X).to_csv('/home/djj/bio/solution/result/multi_fe/X_64.csv', index=False)
pd.DataFrame(Xt).to_csv('/home/djj/bio/solution/result/multi_fe/Xt_64.csv', index=False)



del X, Xt
gc.collect()



X = scipy.sparse.load_npz("/home/djj/bio/solution/data/multimodal-single-cell-as-sparse-matrix/train_multi_inputs_values.sparse.npz")
Xt = scipy.sparse.load_npz("/home/djj/bio/solution/data/multimodal-single-cell-as-sparse-matrix/test_multi_inputs_values.sparse.npz")
both = scipy.sparse.vstack([X, Xt])



TfidfTransformer = tfidfTransformer()
pca = TruncatedSVD(n_components=512, random_state=64)
normalizer = sklearn.preprocessing.Normalizer(norm="l2")

both = TfidfTransformer.fit_transform(both)
both = normalizer.fit_transform(both)
both = np.log1p(both * 1e4)
both = pca.fit_transform(both)
both -= both.mean(axis=1).reshape(-1,1)
both /= both.std(axis=1, ddof=1).reshape(-1,1)
both = both[:,:100]

X = both[:105942]
Xt = both[105942:]
del both
gc.collect()

X0 = pd.read_csv('/home/djj/bio/solution/result/multi_fe/X_64.csv').values
Xt0 = pd.read_csv('/home/djj/bio/solution/result/multi_fe/Xt_64.csv').values

X = np.hstack([X, X0])
Xt = np.hstack([Xt, Xt0])

pd.DataFrame(X).to_csv('/home/djj/bio/solution/result/multi_fe/X_164_l2.csv', index=False)
pd.DataFrame(Xt).to_csv('/home/djj/bio/solution/result/multi_fe/Xt_164_l2.csv', index=False)




