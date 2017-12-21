# -*- coding: utf-8 -*-
import json
import os
import sqlite3
import requests

from flask import Flask, jsonify, request

app = Flask(__name__)

DEBUG = True  # os.environ.get('DEBUG') is not None
VERSION = 0.1

DB_FILENAME = os.environ.get('DB_PATH')
print('Using database', DB_FILENAME)

MANIFESTO_MODEL_HTTP_PORT = os.environ.get('MANIFESTO_MODEL_HTTP_PORT')
print('contact for manifesto model at port', MANIFESTO_MODEL_HTTP_PORT)


@app.route("/", methods=['POST'])
def index():
    return jsonify({})


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

    insert_into(DB_FILENAME, text_ids, labels, 'user')
    n_inserts = len(texts_with_labels)

    # fixme: queue up examples and don't train on every user submission, or contact persistence with manifesto model with scheduler
    training_data = {
        'data': list(map(lambda entry: {'text': entry['statement'], 'label': entry['label']}, get_texts(1000000)))
    }
    url = 'http://manifesto_model:{}/train'.format(MANIFESTO_MODEL_HTTP_PORT)
    print('sending data to manifesto model for training...')
    r = requests.post(
        url=url,
        json=training_data
    )
    print(r.status_code)
    print(r.json())

    return jsonify({'n_inserted': n_inserts}), 201


@app.route("/texts", methods=['GET'])
def texts():
    """
    returns json object like

    {
        'data': [
            {'text_id': 1, 'label': 'left'},
            ...
        ]
    }
    :return:
    """
    # todo: fetch most uncertain texts (select, query model, send n back)
    n_texts = int(request.args.get('n'))
    texts = get_texts(1000000)

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
    print('priotized texts', list(map(lambda e: e['text_id'], priotized_texts)))

    text_data = {'data': priotized_texts}
    return jsonify(text_data)


def insert_into(database_filename, text_ids, labels, annotation_source):
    """
    inserts the texts and labels.

    :param database_filename: name of the sqlite3 database.
    :param text_ids: iterable of int, ids of the texts.
    :param labels: iterable of int.
    :param annotation_source: string, manifesto or user.
    """
    conn = sqlite3.connect(database_filename)
    c = conn.cursor()

    for text_id, label in zip(text_ids, labels):
        c.execute(
            """
            INSERT INTO labels (texts_id, label, source) VALUES
            (?, ?, ?)""",
            (text_id, label, annotation_source)
        )

    conn.commit()
    conn.close()


def get_texts_with_ids(ids):
    """
    :param ids:
    :return:
    """
    print('ids in get texts with ids', ids)
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    return [
        {
            'text_id': text_id,
            'statement': statement
        } for text_id, statement, in c.execute(
            """
            SELECT
                t.id text_id,
                t.statement
            FROM texts t
            WHERE t.id IN({text_ids})""".format(text_ids=','.join('?'*len(ids))),
            ids
        )
    ]


def get_texts(n_texts):
    """
    return at most `n_texts` from the underlying storage.
    :param n_texts: how many texts to retrieve, integer.
    """
    # fixme: 1:n relation of texts to labels, need to implement some kind of majority voting for the labels
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    return [
        {
            'text_id': text_id,
            'statement': statement,
            'label': label,
            'source': source
        } for text_id, statement, label, source, _, _ in c.execute(
            """
            SELECT
                t.id text_id,
                t.statement,
                l.label,
                l.source,
                t.created_at,
                l.created_at
            FROM texts t
            INNER JOIN labels l ON l.texts_id = t.id
            -- ORDER BY RANDOM()
            LIMIT ?""",
            (n_texts, )
        )
    ]


if __name__ == "__main__":
    port = int(os.environ.get("HTTP_PORT"))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=DEBUG,
        use_reloader=False  # with reloader, caused main to be called twice
    )
