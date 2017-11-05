# -*- coding: utf-8 -*-
import flask
from flask import Flask, request, jsonify
import os
from classifier import Classifier, load_data
# from apscheduler.schedulers.background import BackgroundScheduler
import json

app = Flask(__name__)

DEBUG = os.environ.get('DEBUG') != None
VERSION = 0.1

# Schedules news reader to be run at 00:00
#scheduler = BackgroundScheduler()
#scheduler.add_job(retrain, 'interval', minutes=360)
#scheduler.start()

def retrain():
    return Classifier(train=True)

### API
@app.route("/predict", methods=['POST'])
def predict():
    text = request.form['text']
    return jsonify(classifier.predict(text))

@app.route("/get_samples", methods=['POST'])
def get_samples():
    # TODO: wasn't sure how to deal with number of requested samples - should this be a GET?
    n_samples = int(request.form['samples'])
    # TODO: replace data loading with DB data loader
    texts,_ = load_data()
    texts = classifier.prioritize(texts)
    response = {'samples': [{'text':t} for t in texts[:n_samples]]}
    return jsonify(response)

@app.route("/submit_labels", methods=['POST'])
def submit_labels():
    user_id = request.form['user']
    # data = request.form['data']
    # TODO: compute left/right bias
    print(user_id)
    return jsonify({'user_bias': user_id})


if __name__ == "__main__":
    port = 5000
    classifier = Classifier(train=False)
    app.run(host='0.0.0.0', port = port, debug = DEBUG)
