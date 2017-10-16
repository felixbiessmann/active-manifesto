from manifesto_data import get_manifesto_texts
import warnings,json,gzip,re
import os, glob
from scipy.sparse import hstack, vstack
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from scipy.special import logit

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
        parameters = {'alpha': (np.logspace(-7,-4,4)).tolist()}
        # perform gridsearch to get the best regularizer
        gs_clf = GridSearchCV(text_clf, parameters, cv=2, n_jobs=-1,verbose=4)
        gs_clf.fit(X, y)
        report(gs_clf.cv_results_)
    return gs_clf.best_estimator_

def load_data(folder = "data/manifesto"):
    df = pd.concat([pd.read_csv(fn) for fn in glob.glob(os.path.join(folder,"*.csv"))]).dropna(subset=['cmp_code','content'])
    vect = HashingVectorizer()
    return vect.transform(df.content.values), df.cmp_code.apply(int).toarray()

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
    baseline_low = f1_score(y_validation, clf.predict(X_validation), average='weighted')
    clf_trained = SGDClassifier(loss="log",alpha=clf.alpha).fit(vstack([X_train, X_test]), y_train + y_test)
    baseline_high = f1_score(y_validation, clf_trained.predict(X_validation), average='weighted')
    print('\tBaseline on test: {}, baseline score on train and test {}'.format(baseline_low, baseline_high))

    # score test data for active learning sorting
    label_probas = clf.predict_proba(X_test)

    # run a random sampling procedure for training with increasing amounts of labels
    random_priorities = np.random.permutation(label_probas.shape[0])

    random_learning_curve = []
    for percentage in percentage_samples:
        n_samples = int((percentage/100.) * X_test.shape[0])
        X_labelled = X_test[random_priorities[:n_samples],:]
        y_labelled = [y_test[i] for i in random_priorities[:n_samples]]
        try:
            clf_current = SGDClassifier(loss="log",alpha=clf.alpha).fit(vstack([X_train, X_labelled]), y_train + y_labelled)
            current_score = f1_score(y_validation, clf_current.predict(X_validation), average='weighted')
            print('\t(RANDOM) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(n_samples, percentage, current_score, np.round(100.0*(current_score - baseline_low)/(baseline_high-baseline_low))))
            random_learning_curve.append(current_score)
        except:
            pass

    # mean distance to hyperplane
    dists = abs(logit(label_probas)).mean(axis=1)
    # run active learning procedure for training with increasing amounts of labels
    priorities = dists.argsort()

    active_learning_curve = []
    for percentage in percentage_samples:
        n_samples = int((percentage/100.) * X_test.shape[0])
        X_labelled = X_test[priorities[:n_samples],:]
        y_labelled = [y_test[i] for i in priorities[:n_samples]]
        try:
            clf_current = SGDClassifier(loss="log").fit(vstack([X_train, X_labelled]), y_train + y_labelled)
            current_score = f1_score(y_validation, clf_current.predict(X_validation), average='weighted')
            print('\t(ACTIVE LEARNING) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(n_samples, percentage, current_score, np.round(100.0*(current_score - baseline_low)/(baseline_high-baseline_low))))
            active_learning_curve.append(current_score)
        except:
            pass

    return active_learning_curve, random_learning_curve, baseline_low, baseline_high

def run_experiment(test_size=0.9, n_reps=5, percentage_samples=[1,10,50,100]):
    '''
    Runs a multilabel classification experiment
    '''
    print("Loading data")
    X_all, y_all = load_data()

    labels = np.unique(y_all)
    results = {}
    for label in labels[1:]:

        y = [1 if y==label else -1 for y in y_all]

        X_train, X_tolabel, y_train, y_tolabel = train_test_split(X_all, y, test_size=test_size)

        X_test, X_validation, y_test, y_validation = train_test_split(X_tolabel, y_tolabel, test_size=(1-test_size))
        print("Model Selection")
        # do model selection on training data
        clf = model_selection(X_train, y_train)

        # compute active learning curves
        active_learning_curves, random_learning_curves, baseline_lows, baseline_highs = [],[],[],[]
        for irep in range(n_reps):
            X_train, X_tolabel, y_train, y_tolabel = train_test_split(X_all, y, test_size=test_size)
            X_test, X_validation, y_test, y_validation = train_test_split(X_tolabel, y_tolabel, test_size=(1-test_size))
            active_learning_curve, random_learning_curve, baseline_low, baseline_high = compute_active_learning_curve(X_train, y_train, X_test, y_test, X_validation, y_validation, clf,percentage_samples=percentage_samples)
            active_learning_curves.append(active_learning_curve)
            random_learning_curves.append(random_learning_curve)
            baseline_lows.append(baseline_low)
            baseline_highs.append(baseline_high)

            results[label] = {
                'active_learning_curves':active_learning_curves,
                'random_learning_curves':random_learning_curves,
                'baseline_lows':baseline_lows,
                'baseline_highs':baseline_highs,
                'percentage_samples':percentage_samples
                }


    json.dump(results,open("active_learning_curves.json","wt"))
    return active_learning_curves, random_learning_curves, baseline_lows, baseline_highs

def plot_results(fn):
    import pylab
    results = json.load(open(fn))
    for label in results.keys():
        ac = sp.vstack(results[label]['active_learning_curves']).median(axis=1)
        rc = sp.vstack(results[label]['random_learning_curves']).median(axis=1)
        bl = sp.vstack(results[label]['baseline_lows']).median(axis=1)
        bh = sp.vstack(results[label]['baseline_highs']).median(axis=1)
        pylab.figure(figsize=(10,10))
        pylab.hold('all')
        pylab.plot([0.5,.8],[0.5,.8],'k-')
        for i in range(len(results['percentage_samples'])):
            pylab.plot(ac[:,i],rc[:,i],'o')
        pylab.xlim([.64,.7])
        pylab.ylim([.64,.7])
        pylab.legend([0]+results['percentage_samples'])
        pylab.xlabel("Active Learning")
        pylab.ylabel("Random")
        pylab.title("Classifier score as function of n_samples")
        pylab.savefig(label + '.pdf')
