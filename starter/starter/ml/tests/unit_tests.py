"""
Preprocess and Inference Tests
"""

import numpy as np


def test_preprocess_X_shape(X):
    """
    Test the preprocess X output shape.
    """
    assert X.shape[1] == 109


def test_inference_output(preds):
    """
    Test the output of inference module.
    """
    assert np.isin(preds, [1, 0]).all()


def test_target_preprocess_output(target):
    """
    Test modelling target (y) preprocess output.
    """

    assert np.isin(target, [1, 0]).all()
