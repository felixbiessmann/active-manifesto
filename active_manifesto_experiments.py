from manifesto_data import get_manifesto_texts
import warnings,json,gzip,re
import os, glob
from scipy.sparse import hstack, vstack
import scipy as sp
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from scipy.special import logit

label2rightleft = {
    'right': [104,201,203,305,401,402,407,414,505,601,603,605,606],
    'left': [103,105,106,107,403,404,406,412,413,504,506,701,202]
    }

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def model_selection(X,y):
    '''
    Runs model selection, returns fitted classifier
    '''
    # turn off warnings, usually there are some labels missing in the training set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # TODO: pipeline parameters (hashing vectorizer dimensionalities etc.) should also be searchable here
        text_clf = SGDClassifier(loss="log")
        parameters = {'alpha': (np.logspace(-6,-4,5)).tolist()}
        # perform gridsearch to get the best regularizer
        gs_clf = GridSearchCV(text_clf, parameters, cv=2, n_jobs=-1,verbose=4)
        gs_clf.fit(X, y)
        report(gs_clf.cv_results_)
    return gs_clf.best_estimator_

def load_data(folder = "data/manifesto",
        min_label_count = 1000,
        left_right = False):
    df = pd.concat([pd.read_csv(fn) for fn in glob.glob(os.path.join(folder,"*.csv"))]).dropna(subset=['cmp_code','content'])

    if left_right:
        # replace_with = ['left', 'right', 'neutral']
        replace_with = [-1, 1, 0]
        label_to_replace = [
            label2rightleft['left'],
            label2rightleft['right'],
            list(set(df.cmp_code.unique()) - set(label2rightleft['left'] + label2rightleft['right']))
            ]

        for rep, label in zip(replace_with, label_to_replace):
            df.cmp_code.replace(label, rep, inplace = True)

    label_hist = df.cmp_code.value_counts()
    valid_labels = label_hist[label_hist > min_label_count].index
    df = df[df.cmp_code.isin(valid_labels)]
    vect = HashingVectorizer()
    return vect.transform(df.content.values), df.cmp_code.apply(int).as_matrix()

def compute_active_learning_curve(X_train,y_train,X_test,y_test,X_validation, y_validation, clf,percentage_samples=[1,2,5,10,15,30,50,100]):
    '''
    Emulate active learning with annotators:
    for a given training, test and validation set, get the validation error by
    training on training data only, then the score when trained on training and
    test data and then the increasing validation score when adding more labelled
    data, either with random selection or with active learning. The results are
    the increase in scores with the respective sampling policy
    '''
    print('Computing active learning curve:')
    clf = SGDClassifier(loss="log",alpha=clf.alpha).fit(X_train, y_train)
    baseline_low = accuracy_score(y_validation, clf.predict(X_validation))
    clf_trained = SGDClassifier(loss="log",alpha=clf.alpha).fit(vstack([X_train, X_test]), hstack([y_train, y_test]).toarray().flatten())
    baseline_high = accuracy_score(y_validation, clf_trained.predict(X_validation))
    print('\tBaseline on test: {}, baseline score on train and test {}'.format(baseline_low, baseline_high))

    # score test data for active learning sorting
    label_probas = clf.predict_proba(X_test)
    # smoothing probabilities to avoid infs
    eps = 1e-9
    label_probas = np.minimum(label_probas + eps, 1.)

    # run a random sampling procedure for training with increasing amounts of labels
    random_priorities = np.random.permutation(label_probas.shape[0])

    random_learning_curve = []
    for percentage in percentage_samples:
        n_samples = int((percentage/100.) * X_test.shape[0])
        X_labelled = X_test[random_priorities[:n_samples],:]
        y_labelled = y_test[random_priorities[:n_samples]]
        clf_current = SGDClassifier(loss="log",alpha=clf.alpha).fit(vstack([X_train, X_labelled]), hstack([y_train, y_labelled]).toarray().flatten())
        current_score = accuracy_score(y_validation, clf_current.predict(X_validation))
        print('\t(RANDOM) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(n_samples, percentage, current_score, np.round(100.0*(current_score - baseline_low)/(baseline_high-baseline_low))))
        random_learning_curve.append(current_score)

    priorities = prioritize_samples(label_probas)

    active_learning_curve = []
    for percentage in percentage_samples:
        n_samples = int((percentage/100.) * X_test.shape[0])
        X_labelled = X_test[priorities[:n_samples],:]
        y_labelled = y_test[priorities[:n_samples]]

        clf_current = SGDClassifier(loss="log",alpha=clf.alpha).fit(vstack([X_train, X_labelled]), hstack([y_train, y_labelled]).toarray().flatten())
        current_score = accuracy_score(y_validation, clf_current.predict(X_validation))
        # label_probas = clf_current.predict_proba(X_validation)
        # priorities = prioritize_samples(label_probas)
        print('\t(ACTIVE LEARNING) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(n_samples, percentage, current_score, np.round(100.0*(current_score - baseline_low)/(baseline_high-baseline_low))))
        active_learning_curve.append(current_score)

    return active_learning_curve, random_learning_curve, baseline_low, baseline_high

def prioritize_samples(label_probas):
    # mean distance to hyperplane
    # dists = abs(logit(label_probas)).mean(axis=1)
    # priorities = dists.argsort()

    # entropy = -(label_probas * np.log(label_probas)).sum(axis=1)
    # priorities = entropy.argsort()[::-1]

    uncertainty_sampling = 1 - label_probas.max(axis=1)
    priorities = uncertainty_sampling.argsort()[::-1]
    return priorities

def run_experiment(
        baseline_train_percentage=0.1,
        validation_percentage = 0.2,
        n_reps=5,
        percentage_samples=[1,10,20,30,40,50,75,100]):
    '''
    Runs a multilabel classification experiment
    '''
    print("Loading data")
    X_all, y_all = load_data(left_right = True)
    print("Training baseline on {}% (n={})".format(baseline_train_percentage, int(len(y_all) * baseline_train_percentage)))
    print("Validation set size {}% (n={})".format(validation_percentage, int(len(y_all) * validation_percentage)))
    label_pool_percentage = 1-(baseline_train_percentage+validation_percentage)
    print("Label pool {}% (n={})".format(label_pool_percentage,int(len(y_all) * label_pool_percentage)))
    labels = np.unique(y_all)
    results = {}
    X_train, X_tolabel, y_train, y_tolabel = train_test_split(X_all, y_all, test_size=1-baseline_train_percentage)
    X_test, X_validation, y_test, y_validation = train_test_split(X_tolabel, y_tolabel, test_size=validation_percentage / (1-baseline_train_percentage) )
    print("Model Selection")
    # do model selection on training data
    clf = model_selection(X_train, y_train)

    # compute active learning curves
    active_learning_curves, random_learning_curves, baseline_lows, baseline_highs = [],[],[],[]
    for irep in range(n_reps):
        print("Repetition {}/{}".format(irep,n_reps))
        X_train, X_tolabel, y_train, y_tolabel = train_test_split(X_all, y_all, test_size=1-baseline_train_percentage)
        X_test, X_validation, y_test, y_validation = train_test_split(X_tolabel, y_tolabel, test_size=validation_percentage / (1-baseline_train_percentage) )
        active_learning_curve, random_learning_curve, baseline_low, baseline_high = compute_active_learning_curve(X_train, y_train, X_test, y_test, X_validation, y_validation, clf,percentage_samples=percentage_samples)
        active_learning_curves.append(active_learning_curve)
        random_learning_curves.append(random_learning_curve)
        baseline_lows.append(baseline_low)
        baseline_highs.append(baseline_high)

    results = {
        'active_learning_curves':active_learning_curves,
        'random_learning_curves':random_learning_curves,
        'baseline_lows':baseline_lows,
        'baseline_highs':baseline_highs,
        'percentage_samples':percentage_samples
        }

    json.dump(results,open("active_learning_curves.json","wt"))
    return results


def plot_results(fn):
    import pylab
    results = json.load(open(fn))
    ac = sp.median(sp.vstack(results['active_learning_curves']), axis=0)
    rc = sp.median(sp.vstack(results['random_learning_curves']), axis=0)
    bl = sp.median(sp.vstack(results['baseline_lows']), axis=0)
    bh = sp.median(sp.vstack(results['baseline_highs']), axis=0)
    pylab.figure(figsize=(10,6))
    pylab.hold()
    pylab.plot(results['percentage_samples'],rc,'k-o')
    pylab.plot(results['percentage_samples'],ac,'r:o')
    pylab.legend(['random','active'],loc='lower right')
    pylab.title("Classifier score as function of n_samples")
    pylab.xlabel("% samples to label")
    pylab.ylabel("Accuracy")
    pylab.savefig('manuscript/images/active_learning_manifesto.pdf')
