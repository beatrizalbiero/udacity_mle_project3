"""
Preprocess and Inference Tests
"""

import numpy as np
import os.path
import pickle
import sys
import pandas as pd
sys.path.append("..")
import model


model_path = '../clf_model.sav'


def test_compute_model_metrics():
    y_test = pickle.load(open('y.pkl', 'rb'))
    preds = pickle.load(open('preds.pkl', 'rb'))
    p, r, f = model.compute_model_metrics(y_test, preds)
    for metric in [p,r,f]:
        assert isinstance(metric, np.floating)


def test_model_load():
    assert os.path.isfile(model_path) == True
    assert str(type(model.model_load(model_path))) ==\
     "<class 'sklearn.linear_model._logistic.LogisticRegression'>"


def test_inference():
    x_test = pickle.load(open('X_test_preprocessed.pkl', 'rb'))
    clf = model.model_load(model_path)
    pred = model.inference(clf, x_test)
    assert isinstance(pred, np.ndarray)
