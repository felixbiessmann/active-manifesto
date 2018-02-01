# -*- coding: utf-8 -*-
import json
import os
import random
import time
import numpy as np
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from flask import Flask, request, jsonify
from collections import defaultdict
from sqlite_wrapper import get_texts_with_labels, insert_into, get_texts_only, get_texts_with_ids

from classifier import Classifier

app = Flask(__name__)

DEBUG = True  # os.environ.get('DEBUG') is not None
VERSION = 0.1

classifier = Classifier()

n_all_samples = 1000000


def estimate_uncertainty():
    """
    estimates text uncertainties.

    expects a POST-body in the format:
    {
        "data": [
            {"text_id": 17342, "text": "some political statement"},
            ...
        ]
    }

    and returns a possibly truncated list of text ids,
    ordered by uncertainty descending
    {
        "data": [
            {"text_id": 17342},
            ...
        ]
    }
    """
    # n_texts = int(request.args.get('n'))
    request_data = json.loads(request.get_data(as_text=True))['data']
    texts = list(map(lambda entry: entry['text'], request_data))
    text_ids = list(map(lambda entry: entry['text_id'], request_data))

    prios_texts = classifier.prioritize(texts)

    text_ids_priotized = np.array(text_ids)[np.array(prios_texts)]
    response_data = [{"text_id": int(tid)} for tid in text_ids_priotized]
    return jsonify({"data": response_data})


def retrain():
    """
    Fetches latest training data from persistence service and retrains the manifesto model with it.
    """
    response_data = get_texts_with_labels(n_all_samples)
    texts = list(map(lambda entry: entry['statement'], response_data))
    labels = list(map(lambda entry: entry['label'], response_data))
    print('training on', len(texts), 'samples')
    classifier.train(texts, labels)


executors = {
    'default': ThreadPoolExecutor(1),
    'processpool': ProcessPoolExecutor(1)
}
scheduler = BackgroundScheduler(executors=executors)
scheduler.add_job(retrain, 'interval', minutes=60)
scheduler.start()


@app.route("/texts_and_labels", methods=['POST'])
def texts_and_labels():
    """
    Stores labels for the specified text ids.

    expects a POST-body in the format:
    {
        "data": [
            {"text_id": 17342, "label": "left"},
            {"text_id": 873, "label": "neutral"},
            ...
        ]
    }
    """
    #print(request.get_data(as_text=True))
    texts_with_labels = json.loads(request.get_data(as_text=True))['data']
    text_ids = map(lambda entry: entry['text_id'], texts_with_labels)
    labels = map(lambda entry: entry['label'], texts_with_labels)

    insert_into(text_ids, labels, 'user')
    n_inserts = len(texts_with_labels)

    return jsonify({'n_inserted': n_inserts}), 201


@app.route("/prioritized_texts", methods=['GET'])
def prioritized_texts():
    """
    Retrieves all of the political statements from the database, prioritizes them with the manifesto model
    and returns at most as many texts given in the GET parameter `n`, ordered by their uncertainty.

    :return: {
        'data': [
            {'text_id': 1, 'label': 'left'},
            ...
        ]
    }
    """
    print("getting prioritized texts")
    n_texts = int(request.args.get('n'))
    texts = get_texts_only(n_all_samples)
    prios_texts = classifier.prioritize(map(lambda t: t['statement'], texts))
    # print(prios_texts)
    texts_priotized = np.array(texts)[np.array(prios_texts)].tolist()
    # print(texts_priotized)
    return jsonify({'data': texts_priotized[:n_texts]})

@app.route("/prioritized_texts_with_label", methods=['GET'])
def prioritized_texts_with_label():
    """
    Retrieves all of the political statements from the database, prioritizes them with the manifesto model
    and returns at most as many texts given in the GET parameter `n`, ordered by their uncertainty.

    Half of the texts have known labels

    :return: {
        'data': [
            {'text_id': 1, 'label': 'left'},
            ...
        ]
    }
    """
    print("getting prioritized texts")
    n_texts = int(request.args.get('n'))
    texts = get_texts_only(n_all_samples)
    prios_texts = classifier.prioritize(map(lambda t: t['statement'], texts))
    # print(prios_texts)
    texts_prioritized = np.array(texts)[np.array(prios_texts)].tolist()
    to_label = texts_prioritized[:int(n_texts/2)]
    for sample in to_label:
        sample['label'] = ""

    texts_with_labels = random.sample([sample for sample in
        get_texts_with_labels(100, "majority")
        if sample['label'] in ['left', 'right']], int(n_texts/2))

    # print(texts_priotized)
    result = to_label + texts_with_labels
    random.shuffle(result)

    return jsonify({'data': result})


@app.route("/predict", methods=['POST'])
def predict():
    req = json.loads(request.get_data(as_text=True))
    #print('model/predict parsed', req)
    text = req['text']
    result = classifier.predict([text])
    print('result', result)
    return jsonify({'prediction': result})


@app.route("/debug/uncertainties")
def debug_uncertainties():
    """
    Gets all political texts, computes sample uncertainties and groups by label

    :return: sample uncertainties grouped by label {
        'left': [0.1, 0.02, 0.55],
        'neutral': [...],
        ...
    }
    """
    samples = get_texts_with_labels(n_all_samples)
    texts = list(map(lambda sample: sample['statement'], samples))
    labels = list(map(lambda sample: sample['label'], samples))

    probas = classifier.smooth_probas(classifier.clf.predict_proba(texts))
    uncertainties = classifier.per_sample_uncertainty_from(probas).tolist()

    # per label sample uncertainties
    dd = defaultdict(list)
    for label, uncertainty in zip(labels, uncertainties):
        dd[label] = dd[label] + [uncertainty]

    return jsonify(dd)


if __name__ == "__main__":
    retrain()
    port = int(os.environ.get('HTTP_PORT'))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
