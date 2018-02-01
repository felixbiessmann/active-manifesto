from manifesto_data import get_manifesto_texts
import warnings,json,gzip,re
import os, glob
from scipy.sparse import hstack, vstack
import scipy as sp
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.special import logit

label2rightleft = {
    'right': [104,201,203,305,401,402,407,414,505,601,603,605,606],
    'left': [103,105,106,107,403,404,406,412,413,504,506,701,202]
    }

EXPERIMENT_RESULT_FILENAME = "active_learning_curves_50_reps_margin.csv"

def print_classification_report(true,pred,fn='report.txt'):
    s = classfication_report(true,pred) + \
        "\n\n" + "\n".join([",".join(l) for l in confusion_matrix(true,pred)])
    open(fn).write(s)

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
        text_clf = SGDClassifier(loss="log", max_iter=10)
        parameters = {'alpha': (np.logspace(-6,-3,6)).tolist()}
        # perform gridsearch to get the best regularizer
        gs_clf = GridSearchCV(text_clf, parameters, cv=2, n_jobs=-1,verbose=1)
        gs_clf.fit(X, y)
        # report(gs_clf.cv_results_)
    return gs_clf.best_estimator_

def load_data_bow(folder = "../data/manifesto",
        min_label_count = 1000,
        left_right = False,
        max_words = int(1e6)
        ):
    df = pd.concat([pd.read_csv(fn) for fn in glob.glob(os.path.join(folder,"*.csv"))]).dropna(subset=['cmp_code','content'])
    df = df.sample(frac=1.0)
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
    vect = CountVectorizer(max_features=max_words,binary=True).fit(df.content.values)
    return vect.transform(df.content.values), df.cmp_code.apply(int).as_matrix(), vect.vocabulary_, df.content.values, vect

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

def prioritize_samples(label_probas, strategy="margin_sampling", include=[]):
    '''
    Some sampling strategies as in Settles' 2010 Tech Report
    '''
    if len(include) > 0:
        exclude = list(set(range(label_probas.shape[-1]))-set(include))
        excluded_max = label_probas[:, exclude].max(axis=1)
        included = label_probas[:, include]
        included.sort(axis=1)
        included_margin = np.diff(included,axis=1)[:,-1]
        priorities = (included_margin + excluded_max).argsort()
    else:
        if strategy == "entropy_sampling":
            entropy = -(label_probas * np.log(label_probas)).sum(axis=1)
            priorities = entropy.argsort()[::-1]
        elif strategy == "margin_sampling":
            label_probas.sort(axis=1)
            priorities = (label_probas[:,-1] - label_probas[:,-2]).argsort()
        elif strategy == "uncertainty_sampling":
            uncertainty_sampling = 1 - label_probas.max(axis=1)
            priorities = uncertainty_sampling.argsort()[::-1]
        elif strategy == "random":
            priorities = np.random.permutation(label_probas.shape[0])

    return priorities

def compute_active_learning_curve(
    X_tolabel,
    y_tolabel,
    X_validation,
    y_validation,
    percentage_samples=[1,2,5,10,15,30,50,75,100],
    strategies = ["random", 'margin_sampling']):
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

    learning_curves = []
    for strategy in strategies:
        # initially use random priorities, as we don't have a trained model
        priorities = np.random.permutation(X_tolabel.shape[0]).tolist()
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
            print('\t(Strategy {}) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(strategy, n_training_samples, percentage, current_score, np.round(100.0*(current_score - learning_curves[0]['score'])/(baseline_high-learning_curves[0]['score']))))

            if len(to_label) > 0:
                # prioritize the not yet labeled data points
                priorities = prioritize_samples(clf_current.predict_proba(X_tolabel[to_label,:]), strategy)
                # get indices in original data set for not yet labeled data
                priorities = [to_label[idx] for idx in priorities]


    return pd.DataFrame(learning_curves)

def compute_informed_active_learning_curve(
    X_tolabel,
    y_tolabel,
    X_validation,
    y_validation,
    percentage_samples=[1,30,50,75,100],
    strategies = ['random', 'margin_sampling', 'informed'],
    include_labels = [-1,1]):
    '''
    Emulate active learning with annotators, but neglect some classes during sampling:

    '''

    def evaluate(y,yhat,included=[-1,0,1]):
        true, predicted = zip(*[(t,p) for t,p in zip(y, yhat) if t in included])
        return accuracy_score(true, predicted)

    labels = np.unique(y_tolabel)
    include = [idx for idx, label in enumerate(labels) if label in include_labels]
    exclude = [idx for idx, label in enumerate(labels) if label not in include_labels]

    print('Computing active learning curve:')
    clf_trained = model_selection(X_tolabel, y_tolabel)
    baseline_high = evaluate(y_validation, clf_trained.predict(X_validation))
    print('\tBaseline on 100% of data {}'.format(baseline_high))

    learning_curves = []

    for strategy in strategies:
        # initially use random priorities, as we don't have a trained model
        priorities = np.random.permutation(X_tolabel.shape[0]).tolist()
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
            y_validation_predicted = clf_current.predict(X_validation)
            current_score = evaluate(y_validation, y_validation_predicted)
            learning_curves.append(
                    {
                    'percentage_samples': percentage,
                    'strategy': strategy,
                    'score': current_score,
                    'confusion_matrix': confusion_matrix(y_validation, y_validation_predicted)
                    }
                )
            print('\t(Stragety {}) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(strategy, n_training_samples, percentage, current_score, np.round(100.0*(current_score - learning_curves[0]['score'])/(baseline_high-learning_curves[0]['score']))))

            if len(to_label) > 0:
                # prioritize the not yet labeled data points
                if strategy=='informed':
                    priorities = prioritize_samples(clf_current.predict_proba(X_tolabel[to_label,:]), include=include)
                else:
                    priorities = prioritize_samples(clf_current.predict_proba(X_tolabel[to_label,:]), strategy)
                # get indices in original data set for not yet labeled data
                priorities = [to_label[idx] for idx in priorities]

    return pd.DataFrame(learning_curves)

def run_explanations_experiment(validation_percentage = 0.1, top=5):
    def rem_words(row):
        for w in row['rel_words_pos']:
            row['newtexts'] = row['newtexts'].lower().replace(w,"")
        return row
    X_all, y_all, vocab, texts, vect = load_data_bow(left_right = True)
    idx2word = {v:k for k,v in vocab.items()}
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=validation_percentage,shuffle=False)
    clf = model_selection(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    y_test_bin = np.zeros((len(y_test),len(clf.classes_)))
    for idx,y_test_i in enumerate(y_test):
        y_test_bin[idx,y_test_i] = 1
    errors = y_test_bin - y_pred
    errors[:,1] = 0
    gradient = errors.dot(clf.coef_)
    # rel_words = gradient.argsort(axis=1)
    rel_words_neg = [[idx2word[widx] for widx in t.argsort()[:top]] for n,t in enumerate(gradient)]
    rel_words_pos = [[idx2word[widx] for widx in t.argsort()[-top:][::-1]] for n,t in enumerate(gradient)]
    df = pd.DataFrame({
        'true':y_test,
        'pred':clf.predict(X_test),
        'texts':texts[-len(y_test):],
        'rel_words_pos':rel_words_pos,
        'rel_words_neg':rel_words_neg
        })

    df['newtexts'] = df['texts']
    df = df.apply(rem_words,axis=1)
    df['newtexts'] = df['newtexts'] + df['rel_words_neg'].apply(lambda x: " ".join(x))

    df['new_predictions'] = clf.predict(vect.transform(df.newtexts.tolist()))

    return df[['true','pred','new_predictions','rel_words_pos','rel_words_neg','texts','newtexts']]

def run_experiment_informed(
        validation_percentage = 0.1,
        n_reps=100,
        percentage_samples=[1,50,60,70,80,90,100],
        output_filename="informed_active_learning_curves.csv"):
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
        learning_curve = compute_informed_active_learning_curve(
                X_tolabel, y_tolabel,
                X_validation, y_validation,
                percentage_samples=sp.unique(percentage_samples))
        learning_curve['repetition'] = irep
        learning_curves.append(learning_curve)

    pd.concat(learning_curves).to_csv(output_filename)


def run_experiment(
        validation_percentage = 0.1,
        n_reps=100,
        percentage_samples=[1,10,20,30,40,50,60,70,80,90,100],
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
                percentage_samples=sp.unique(percentage_samples))
        learning_curve['repetition'] = irep
        learning_curves.append(learning_curve)

    pd.concat(learning_curves).to_csv(output_filename)

def run_baseline(validation_percentage = 0.5):
    '''
    Runs experiment on all data and prints classification report
    '''
    print("Loading data")
    X_all, y_all = load_data(left_right = False)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=validation_percentage)
    clf = model_selection(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred).replace("     ",'&').replace('\n','\\\\\n'))

def plot_results(fn=EXPERIMENT_RESULT_FILENAME):
    import seaborn, pylab

    df = pd.read_csv(fn)
    pylab.figure(figsize=(12,6))
    seaborn.set(font_scale=2)
    seaborn.set_style('whitegrid')
    pylab.hold('all')
    linestyles = zip(df.strategy.unique(),[':','-.','--','-'])
    for strategy,linestyle in linestyles:
        axes = seaborn.tsplot(
            time="percentage_samples",
            value="score",
            condition="strategy",
            unit="repetition",
            err_style="ci_bars",
            ci=[.05,95],
            lw=2,
            data=df[df.strategy==strategy], estimator=np.median, linestyle=linestyle)
    pylab.title('Left vs. Neutral vs. Right')
    pylab.xlim([10,101])
    pylab.ylim([0.55,0.65])
    pylab.ylabel("Accuracy")
    pylab.xlabel("Labeling budget (% of total training data)")
    pylab.tight_layout()
    pylab.savefig('../manuscript/images/active_learning_manifesto.pdf')

def plot_informed_active_learning(fn="informed_active_learning_curves.csv"):
    import seaborn, pylab, re
    df = pd.read_csv(fn)
    def get_cm(row):
        return np.array([[int(y) for y in re.findall(r'\d+', x)] for x in row.split("\n")])

    def get_acc_from_cm(cm):
        cm_ex = cm[[0,2],:][:,[0,2]]
        return np.diag(cm_ex).sum() / cm_ex.sum()

    df['binary_acc'] = df.confusion_matrix.apply(get_cm).apply(get_acc_from_cm)


    pylab.figure(figsize=(12,6))
    seaborn.set(font_scale=2)
    seaborn.set_style('whitegrid')

    axes = seaborn.tsplot(
        time="percentage_samples",
        value="score",
        condition="strategy",
        unit="repetition",
        err_style="ci_bars",
        ci=[.05,95],
        lw=2,
        data=df, estimator=np.median)
    pylab.xlim([49,101])
    pylab.ylim([0.6,0.65])

    pylab.title('Left vs Right vs Neutral')
    pylab.ylabel("Accuracy")
    pylab.xlabel("Labeling budget (% of total training data)")
    pylab.tight_layout()
    pylab.savefig('../manuscript/images/active_learning_manifesto_informed_three_class.pdf')

    pylab.figure(figsize=(12,6))
    seaborn.set(font_scale=2)
    seaborn.set_style('whitegrid')

    axes = seaborn.tsplot(
        time="percentage_samples",
        value="binary_acc",
        condition="strategy",
        unit="repetition",
        err_style="ci_bars",
        ci=[.05,95],
        lw=2,
        data=df, estimator=np.median)
    pylab.xlim([49,101])
    pylab.ylim([0.795,0.83])
    pylab.title('Left vs. Right')
    pylab.ylabel("Accuracy")
    pylab.xlabel("Labeling budget (% of total training data)")
    pylab.tight_layout()
    pylab.savefig('../manuscript/images/active_learning_manifesto_informed_two_class.pdf')

def plot_label_histogram(folder = "../data/manifesto"):
    manifesto_labels = pd.read_csv(os.path.join(folder,"manifestolabels.txt"),sep=" ",names=['cmp_code','label'])
    manifesto_labels['cmp_code'] = manifesto_labels.cmp_code.apply(lambda x: int(x[3:]))
    df = pd.concat([pd.read_csv(fn) for fn in glob.glob(os.path.join(folder,"*.csv"))]).dropna(subset=['cmp_code','content'])
    counts = df.cmp_code.value_counts()
    count_df = manifesto_labels.join(counts,on='cmp_code',how='inner',lsuffix="_l").sort_values(by='cmp_code',ascending=False)
    count_df.columns = ['cmp_code', 'label', 'counts']

    print(count_df[count_df.counts>1000].to_latex(index=False))



if __name__ == "__main__":
    run_experiment()
    plot_results()
