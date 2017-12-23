# -*- coding: utf-8 -*-
import json
import os
import requests

from sqlite_wrapper import get_texts_with_labels, insert_into, get_texts_only, get_texts_with_ids

from flask import Flask, jsonify, request

app = Flask(__name__)

DEBUG = True  # os.environ.get('DEBUG') is not None
VERSION = 0.1

MANIFESTO_MODEL_HTTP_PORT = os.environ.get('MANIFESTO_MODEL_HTTP_PORT')
print('contact for manifesto model at port', MANIFESTO_MODEL_HTTP_PORT)


@app.route("/", methods=['POST'])
def index():
    return jsonify({})


@app.route("/training_texts", methods=['GET'])
def training_texts():
    """
    Endpoint that returns the current state of the training data.
    :return: {
        'data': [
            {],
            ...
        ]
    }
    """
    texts = get_texts_with_labels(1000000)
    texts = list(map(lambda entry: {'text': entry['statement'], 'label': entry['label']}, texts))
    return jsonify({'data': texts}), 200


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
    texts_with_labels = json.loads(request.get_data(as_text=True))['data']
    text_ids = map(lambda entry: entry['text_id'], texts_with_labels)
    labels = map(lambda entry: entry['label'], texts_with_labels)

    insert_into(text_ids, labels, 'user')
    n_inserts = len(texts_with_labels)

    return jsonify({'n_inserted': n_inserts}), 201


@app.route("/prioritized_texts", methods=['GET'])
def prioritized_texts():
    """
    Retrieves all of the political statements from the database, sends them to the manifesto model to be prioritized
    and returns at most as many texts given in the GET parameter `n`, ordered by their uncertainty.

    :return: {
        'data': [
            {'text_id': 1, 'label': 'left'},
            ...
        ]
    }
    """
    n_texts = int(request.args.get('n'))
    texts = get_texts_only(1000000)

    samples = {
        'data': list(map(lambda entry: {'text_id': entry['text_id'], 'text': entry['statement']}, texts))
    }

    # ask model about sample uncertainty
    url = 'http://manifesto_model:{}/estimate_uncertainty'.format(MANIFESTO_MODEL_HTTP_PORT)
    r = requests.post(
        url=url,
        json=samples
    )
    priotized_text_ids = list(map(lambda e: int(e['text_id']), r.json()['data'][:n_texts]))
    priotized_texts = get_texts_with_ids(priotized_text_ids)

    text_data = {'data': priotized_texts}
    return jsonify(text_data)


if __name__ == "__main__":
    port = int(os.environ.get("HTTP_PORT"))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=DEBUG,
        use_reloader=False  # with reloader, caused main to be called twice
    )
