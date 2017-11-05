# -*- coding: utf-8 -*-
import pickle, json, os, glob, sys
import scipy as sp
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.externals import joblib

# manifestoproject codes for left/right orientation
label2rightleft = {
    'right': [104,201,203,305,401,402,407,414,505,601,603,605,606],
    'left': [103,105,106,107,403,404,406,412,413,504,506,701,202]
    }

# FIXME: load_data should be replaced with loading from DB
def load_data(folder = "../../data/manifesto",
        min_label_count = 1000):
    df = pd.concat([pd.read_csv(fn) for fn in glob.glob(os.path.join(folder,"*.csv"))]).dropna(subset=['cmp_code','content'])
    replace_with = ['left', 'right', 'neutral']
    label_to_replace = [
        label2rightleft['left'],
        label2rightleft['right'],
        list(set(df.cmp_code.unique()) - set(label2rightleft['left'] + label2rightleft['right']))
        ]

    for rep, label in zip(replace_with, label_to_replace):
        df.cmp_code.replace(label, rep, inplace = True)

    return df.content.values, df.cmp_code

class Classifier:

    def __init__(self,train=False):
        '''
        Creates a classifier object
        if no model is found, or train is set True, a new classifier is learned

        INPUT
        folder  the root folder with the raw text data, where the model is stored
        train   set True if you want to train

        '''
        # if there is no classifier file or training is invoked
        if (not os.path.isfile('classifier.pickle')) or train:
            print('Training classifier')
            self.train()
        print('Loading classifier')
        self.clf = joblib.load('classifier.pickle')

    def predict(self,text):
        '''
        Uses scikit-learn Bag-of-Word extractor and classifier and
        applies it to some text.

        INPUT
        text    a string to assign to a manifestoproject label

        '''
        # make it a list, if it is a string
        if not type(text) is list: text = [text]
        # predict probabilities
        probabilities = self.clf.predict_proba(text).flatten()
        predictions = dict(zip(self.clf.steps[-1][1].classes_, probabilities.tolist()))

        # transform the predictions into json output
        return predictions

    def train(self):
        '''
        trains a classifier on the bag of word vectors

        INPUT
        folds   number of cross-validation folds for model selection

        '''
        # the scikit learn pipeline for vectorizing, normalizing and classifying text
        text_clf = Pipeline([('vect', HashingVectorizer()),
                            ('clf',SGDClassifier(loss="log", max_iter=3))])

        parameters = {'clf__alpha': (10.**sp.arange(-6,-4,1.)).tolist()}
        # load the data
        data,labels = load_data()
        # perform gridsearch to get the best regularizer
        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1,verbose=4)
        gs_clf.fit(data,labels)
        # dump classifier to pickle
        joblib.dump(gs_clf.best_estimator_, 'classifier.pickle')
