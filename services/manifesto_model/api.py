# -*- coding: utf-8 -*-
import json
import os

import numpy as np
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, request, jsonify

from classifier import Classifier

app = Flask(__name__)

DEBUG = True  # os.environ.get('DEBUG') != None
VERSION = 0.1

PERSISTENCE_HTTP_PORT = int(os.environ.get('PERSISTENCE_HTTP_PORT'))


def retrain():
    """
    Fetches latest training data from persistence service and retrains the manifesto model with it.
    """
    print('retrain called')

    url = 'http://persistence:{}/training_texts'.format(PERSISTENCE_HTTP_PORT)
    print('requesting training data from persistence...')
    r = requests.get(url=url)
    print(r.status_code)

    response_data = r.json()['data']
    texts = list(map(lambda entry: entry['text'], response_data))
    labels = list(map(lambda entry: entry['label'], response_data))

    print('training on', len(texts), 'samples')
    classifier.train(texts, labels)


scheduler = BackgroundScheduler()
scheduler.add_job(retrain, 'interval', minutes=60)
scheduler.start()


@app.route("/predict", methods=['POST'])
def predict():
    text = request.form['text']
    return jsonify(classifier.predict(text))


@app.route("/estimate_uncertainty", methods=['POST'])
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
    request_data = json.loads(request.get_data(as_text=True))['data']
    texts = list(map(lambda entry: entry['text'], request_data))
    text_ids = list(map(lambda entry: entry['text_id'], request_data))

    prios_texts = classifier.prioritize(texts)

    text_ids_priotized = np.array(text_ids)[np.array(prios_texts)]
    response_data = [{"text_id": int(tid)} for tid in text_ids_priotized]
    return jsonify({"data": response_data})


if __name__ == "__main__":
    port = int(os.environ.get('HTTP_PORT'))
    classifier = Classifier(train=False)
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
