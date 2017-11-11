# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# manifestoproject codes for left/right orientation
label2rightleft = {
    'right': [104, 201, 203, 305, 401, 402, 407, 414, 505, 601, 603, 605, 606],
    'left': [103, 105, 106, 107, 403, 404, 406, 412, 413, 504, 506, 701, 202]
}


class Classifier:
    def __init__(self, train=False):
        """
        Creates a classifier object
        if no model is found, or train is set True, a new classifier is learned

        INPUT
        folder  the root folder with the raw text data, where the model is stored
        train   set True if you want to train

        """
        self.clf = None

    @staticmethod
    def smooth_probas(label_probas, eps=1e-9):
        # smoothing probabilities to avoid infs
        return sp.minimum(label_probas + eps, 1.)

    def prioritize(self, texts, strategy='margin_sampling'):
        """
        Some sampling strategies as in Settles' 2010 Tech Report
        """
        label_probas = self.smooth_probas(self.clf.predict_proba(texts))
        if strategy == "entropy_sampling":
            entropy = -(label_probas * np.log(label_probas)).sum(axis=1)
            priorities = entropy.argsort()[::-1]
        elif strategy == "margin_sampling":
            label_probas.sort(axis=1)
            priorities = (label_probas[:, -1] - label_probas[:, -2]).argsort()
        elif strategy == "uncertainty_sampling":
            uncertainty_sampling = 1 - label_probas.max(axis=1)
            priorities = uncertainty_sampling.argsort()[::-1]

        return priorities

    def predict(self, text):
        """
        Uses scikit-learn Bag-of-Word extractor and classifier and
        applies it to some text.

        INPUT
        text    a string to assign to a manifestoproject label

        """
        # make it a list, if it is a string
        if not type(text) is list: text = [text]
        # predict probabilities
        probabilities = self.clf.predict_proba(text).flatten()
        predictions = dict(zip(self.clf.steps[-1][1].classes_, probabilities.tolist()))

        # transform the predictions into json output
        return predictions

    def train(self, data, labels):
        """
        trains a classifier on the bag of word vectors

        INPUT
        folds   number of cross-validation folds for model selection
        """
        # the scikit learn pipeline for vectorizing, normalizing and classifying text
        text_clf = Pipeline([('vect', HashingVectorizer()),
                             ('clf', SGDClassifier(loss="log", max_iter=3))])

        parameters = {'clf__alpha': (10. ** sp.arange(-6, -4, 1.)).tolist()}
        # perform gridsearch to get the best regularizer
        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, verbose=4)
        gs_clf.fit(data, labels)
        # dump classifier to pickle
        joblib.dump(gs_clf.best_estimator_, 'classifier.pickle')

        self.clf = gs_clf.best_estimator_
