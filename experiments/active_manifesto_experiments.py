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

EXPERIMENT_RESULT_FILENAME = "active_learning_curves.csv"

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
        parameters = {'alpha': (np.logspace(-6,-3,6)).tolist()}
        # perform gridsearch to get the best regularizer
        gs_clf = GridSearchCV(text_clf, parameters, cv=2, n_jobs=-1,verbose=4)
        gs_clf.fit(X, y)
        report(gs_clf.cv_results_)
    return gs_clf.best_estimator_

def load_data(folder = "../data/manifesto",
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

def smooth_probas(label_probas, eps = 1e-9):
    # smoothing probabilities to avoid infs
    return np.minimum(label_probas + eps, 1.)

def prioritize_samples(label_probas, strategy="margin_sampling"):
    '''
    Some sampling strategies as in Settles' 2010 Tech Report
    '''
    if strategy == "entropy_sampling":
        entropy = -(label_probas * np.log(label_probas)).sum(axis=1)
        priorities = entropy.argsort()[::-1]
    elif strategy == "margin_sampling":
        label_probas.sort(axis=1)
        priorities = (label_probas[:,-1] - label_probas[:,-2]).argsort()
    elif strategy == "uncertainty_sampling":
        uncertainty_sampling = 1 - label_probas.max(axis=1)
        priorities = uncertainty_sampling.argsort()[::-1]

    return priorities

def compute_active_learning_curve(
    X_tolabel,
    y_tolabel,
    X_validation,
    y_validation,
    percentage_samples=[1,2,5,10,15,30,50,75,100],
    strategies = ['margin_sampling']):
    '''
    Emulate active learning with annotators:
    for a given training, test and validation set, get the validation error by
    training on training data only, then the score when trained on training and
    test data and then the increasing validation score when adding more labelled
    data, either with random selection or with active learning. The results are
    the increase in scores with the respective sampling policy
    '''
    print('Computing active learning curve:')
    clf_trained = model_selection(X_tolabel, y_tolabel)
    baseline_high = accuracy_score(y_validation, clf_trained.predict(X_validation))
    print('\tBaseline on 100% of data {}'.format(baseline_high))

    # run a random sampling procedure for training with increasing amounts of labels
    random_priorities = np.random.permutation(X_tolabel.shape[0])
    learning_curves = []

    for percentage in percentage_samples:
        n_samples = int((percentage/100.) * X_tolabel.shape[0])
        X_labelled = X_tolabel[random_priorities[:n_samples],:]
        y_labelled = y_tolabel[random_priorities[:n_samples]]
        clf_current = model_selection(X_labelled, y_labelled)
        current_score = accuracy_score(y_validation, clf_current.predict(X_validation))
        learning_curves.append(
                {
                'percentage_samples': percentage,
                'strategy': 'random',
                'score': current_score
                }
            )
        print('\t(RANDOM) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(n_samples, percentage, current_score, np.round(100.0*(current_score - learning_curves[0]['score'])/(baseline_high-learning_curves[0]['score']))))

    for strategy in strategies:
        # initially use random priorities, as we don't have a trained model
        priorities = random_priorities.tolist()
        N = X_tolabel.shape[0]
        all_indices = set(range(N))
        labelled = []
        for percentage in percentage_samples:
            n_training_samples = int((percentage/100.) * N) - len(labelled)
            labelled += priorities[:n_training_samples]
            X_labelled = X_tolabel[labelled,:]
            y_labelled = y_tolabel[labelled]
            clf_current = model_selection(X_labelled, y_labelled)

            # get the not yet labeled data point indices
            to_label = list(all_indices - set(labelled))
            current_score = accuracy_score(y_validation, clf_current.predict(X_validation))
            learning_curves.append(
                    {
                    'percentage_samples': percentage,
                    'strategy': strategy,
                    'score': current_score
                    }
                )
            print('\t(ACTIVE {}) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(strategy, n_training_samples, percentage, current_score, np.round(100.0*(current_score - learning_curves[0]['score'])/(baseline_high-learning_curves[0]['score']))))

            if len(to_label) > 0:
                # prioritize the not yet labeled data points
                priorities = prioritize_samples(clf_current.predict_proba(X_tolabel[to_label,:]), strategy)
                # get indices in original data set for not yet labeled data
                priorities = [to_label[idx] for idx in priorities]


    return pd.DataFrame(learning_curves)

def run_experiment(
        validation_percentage = 0.2,
        n_reps=50,
        percentage_samples=[20,30,40,50,50,70,80,90,100],
        output_filename=EXPERIMENT_RESULT_FILENAME):
    '''
    Runs a multilabel classification experiment
    '''
    print("Loading data")
    X_all, y_all = load_data(left_right = True)
    print("Validation set size {}% (n={})".format(validation_percentage, int(len(y_all) * validation_percentage)))
    label_pool_percentage = 1 - validation_percentage
    print("Label pool {}% (n={})".format(label_pool_percentage,int(len(y_all) * label_pool_percentage)))
    labels = np.unique(y_all)
    results = {}

    # compute active learning curves
    learning_curves = []
    for irep in range(n_reps):
        print("Repetition {}/{}".format(irep,n_reps))
        X_tolabel, X_validation, y_tolabel, y_validation = \
            train_test_split(X_all, y_all, test_size=validation_percentage)
        learning_curve = compute_active_learning_curve(
                X_tolabel, y_tolabel,
                X_validation, y_validation,
                percentage_samples=percentage_samples)
        learning_curve['repetition'] = irep
        learning_curves.append(learning_curve)

    results = pd.concat(learning_curves).to_csv(output_filename, index = False)

def plot_results(fn=EXPERIMENT_RESULT_FILENAME):
    import seaborn, pylab
    df = pd.read_csv(fn)
    pylab.figure()
    seaborn.tsplot(
        time="percentage_samples",
        value="score",
        condition="strategy",
        unit="repetition",
        err_style="ci_bars",
        ci=95,
        lw=1,
        data=df)
    pylab.title('prioritization strategy performances (95% CI shown)')
    pylab.savefig('../manuscript/images/active_learning_manifesto.pdf')

if __name__ == "__main__":
    run_experiment()
    plot_results()
