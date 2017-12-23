# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from sklearn.externals import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import scipy.stats


# manifestoproject codes for left/right orientation
label2rightleft = {
    'right': [104, 201, 203, 305, 401, 402, 407, 414, 505, 601, 603, 605, 606],
    'left': [103, 105, 106, 107, 403, 404, 406, 412, 413, 504, 506, 701, 202]
}


class Classifier:
    def __init__(self):
        """
        Creates a classifier object
        if no model is found, or train is set True, a new classifier is learned

        INPUT
        folder  the root folder with the raw text data, where the model is stored
        train   set True if you want to train

        """
        self.clf_path = 'classifier.pickle'
        self.clf = None
        try:
            self.clf = joblib.load(self.clf_path)
        except:
            pass

    @staticmethod
    def smooth_probas(label_probas, eps=1e-9):
        # smoothing probabilities to avoid infs
        return sp.minimum(label_probas + eps, 1.)

    @staticmethod
    def per_sample_uncertainty_from(label_probas, strategy='margin_sampling'):
        """
        :param texts:
        :return:
        """
        if strategy == "entropy_sampling":
            entropy = -(label_probas * np.log(label_probas)).sum(axis=1)
            return entropy
        elif strategy == "margin_sampling":
            label_probas.sort(axis=1)
            return label_probas[:, -1] - label_probas[:, -2]
        elif strategy == "uncertainty_sampling":
            uncertainty_sampling = 1 - label_probas.max(axis=1)
            return uncertainty_sampling

    def prioritize(self, texts, strategy='margin_sampling'):
        """
        Some sampling strategies as in Settles' 2010 Tech Report
        """

        label_probas = self.smooth_probas(self.clf.predict_proba(texts))

        uncertainty_measure = self.per_sample_uncertainty_from(label_probas, strategy)

        if strategy == "entropy_sampling":
            priorities = uncertainty_measure.argsort()[::-1]
        elif strategy == "margin_sampling":
            priorities = uncertainty_measure.argsort()
        elif strategy == "uncertainty_sampling":
            priorities = uncertainty_measure.argsort()[::-1]

        return priorities

    def predict(self, text):
        """
        Uses scikit-learn Bag-of-Word extractor and classifier and
        applies it to some text.

        INPUT
        text    a string to assign to a manifestoproject label

        """
        # make it a list, if it is a string
        # if not type(text) is list: text = [text]
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
        text_clf = Pipeline([('vect', HashingVectorizer()), ('clf', SGDClassifier(loss="log", max_iter=3))])
        parameters = {'clf__alpha': (10. ** sp.arange(-6, -4, 1.)).tolist()}
        # perform gridsearch to get the best regularizer
        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, verbose=4)
        gs_clf.fit(data, labels)
        # dump classifier to pickle
        joblib.dump(gs_clf.best_estimator_, self.clf_path)

        self.clf = gs_clf.best_estimator_


# if __name__ == "__main__":
#     def feature_uncertainties(texts):
#         """
#         :param texts: n-element list of string, containing the to be uncertainty estimated political statements
#         :return:
#         """
#         clf = Classifier()
#         clf_class_probabilities = clf.clf.predict_proba(texts)
#         sample_per_class_probas = [dict(zip(clf.clf.steps[-1][1].classes_, class_probas)) for class_probas in clf_class_probabilities.tolist()]
#         for p in sample_per_class_probas:
#             print(p)
#
#         X = clf.clf.steps[0][1].transform(texts)
#         df = pd.DataFrame(X.toarray())
#         # N x D, for N samples with dimensionality D each
#         print(df.shape)
#         smoothed = clf.smooth_probas(clf_class_probabilities)
#         per_sample_uncertainty = clf.per_sample_uncertainty_from(smoothed, strategy='entropy_sampling')
#         print(per_sample_uncertainty)
#         df = pd.concat((df, pd.DataFrame({'sample_uncertainty': per_sample_uncertainty})), axis=1)
#         # df.corr() OOM
#         # df mostly zero
#         # print(df.sample_uncertainty)
#         # print(df.loc[:, 34].corr(df.loc[:, 'sample_uncertainty']))
#         for idx in range(df.shape[1]):
#             r, p = scipy.stats.pearsonr(df.iloc[:, idx].values, df.sample_uncertainty)
#             if p < 1.0:
#                 print(idx, r, p)
#
#         # print(df.loc[:, 0])
#         # print(df.loc[:, 'sample_uncertainty'])
#
#
#     texts = [
#         "krankenkassen mehr beitrag",
#         "das ist eine poltische entscheidung gesellschaft verantwortung",
#         "wille zur allgemeinen sicherheit in deutschland"
#     ]
#
#     feature_uncertainties(texts)
