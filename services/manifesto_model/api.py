# -*- coding: utf-8 -*-
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

if __name__ == "__main__":
    port = 5000
    classifier = retrain()
    app.run(host='0.0.0.0', port = port, debug = DEBUG)
