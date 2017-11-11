# -*- coding: utf-8 -*-
import numpy as np
import flask
from flask import Flask, request, jsonify
import os
from classifier import Classifier
# from apscheduler.schedulers.background import BackgroundScheduler
import json

app = Flask(__name__)

DEBUG = os.environ.get('DEBUG') != None
VERSION = 0.1


# Schedules news reader to be run at 00:00
# scheduler = BackgroundScheduler()
# scheduler.add_job(retrain, 'interval', minutes=360)
# scheduler.start()

def retrain():
    return Classifier(train=True)


### API
@app.route("/predict", methods=['POST'])
def predict():
    text = request.form['text']
    return jsonify(classifier.predict(text))


@app.route("/train", methods=['POST'])
def train():
    """
    trains a classifier.

    expects a POST-body in the format:
    {
        "data": [
            {"text": "some political statement", "label": "left"},
            ...
        ]
    }
    """
    request_data = json.loads(request.get_data(as_text=True))['data']
    texts = list(map(lambda entry: entry['text'], request_data))
    labels = list(map(lambda entry: entry['label'], request_data))

    classifier.train(texts, labels)

    return jsonify({}), 201


@app.route("/estimate_uncertainty", methods=['POST'])
def get_samples():
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
    # port = int(os.environ.get('HTTP_PORT'))
    port = int(os.environ.get('HTTP_PORT'))
    classifier = Classifier(train=False)
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
