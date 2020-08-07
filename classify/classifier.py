"""
Implement classifier for ln_matcher.

This module defines an interface for classifiers.

Todo:
    * Use column context info. Implement more relevant classifiers.
    * Implement simple rules (e.g., does data match a phone number?).
    * Shuffle data before k-fold cutting in predict_training.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from abc import ABCMeta, abstractmethod


class Classifier(object):

    """Define classifier interface for ln_matcher."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, data):
        """Initialize the class."""
        pass

    @abstractmethod
    def fit(self, data):
        """Train based on the input training data."""
        pass

    @abstractmethod
    def predict_training(self, folds):
        """Predict the training data (using k-fold cross validation)."""
        pass

    @abstractmethod
    def predict(self, data):
        """Predict for unseen data."""
        pass
