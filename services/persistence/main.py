# -*- coding: utf-8 -*-
import json
import os
import sqlite3

from flask import Flask, jsonify, request

app = Flask(__name__)

DEBUG = True  # os.environ.get('DEBUG') is not None
VERSION = 0.1

DB_FILENAME = os.environ.get('DB_PATH')
print('Using database', DB_FILENAME)


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

    return jsonify({'n_inserted': n_inserts}), 201


@app.route("/texts", methods=['GET'])
def texts():
    n_texts = request.args.get('n')
    text_data = {'data': get_texts(n_texts)}
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


def get_texts(n_texts):
    """
    return at most `n_texts` from the underlying storage.
    :param n_texts: how many texts to retrieve, integer.
    """
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
        ORDER BY RANDOM()
        LIMIT ?""",
        (n_texts, )
        )
    ]


def create_manifesto_storage(texts, labels):
    """
    creates required tables and inserts all of the given texts and labels.

    :param texts: iterable of political texts.
    :param labels: iterable of political text labels.
    """
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS texts')
    c.execute(
        '''CREATE TABLE texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                statement TEXT NOT NULL
            )
        '''
    )
    c.execute('DROP TABLE IF EXISTS labels')
    c.execute(
        '''CREATE TABLE labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                texts_id INTEGER NOT NULL,
                label INTEGER NOT NULL,
                source text NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(texts_id) REFERENCES texts(id)
            )
        '''
    )
    conn.commit()
    conn.close()

    insert_into(DB_FILENAME, texts, labels, 'manifesto')

    # for statement, label, source, text_created, label_created in c.execute(
    #         """
    #         SELECT
    #             t.statement,
    #             l.label,
    #             l.source,
    #             t.created_at,
    #             l.created_at
    #         FROM texts t
    #         INNER JOIN labels l ON l.texts_id = t.id
    #         LIMIT 10"""
    # ):
    #     print(statement, label, source)
    #
    # print(
    #     'inserted',
    #     c.execute("SELECT count(1) FROM texts").fetchone(),
    #     'texts with',
    #     c.execute("SELECT count(1) FROM labels").fetchone(),
    #     'labels'
    # )
    #
    # conn.close()


if __name__ == "__main__":
    port = int(os.environ.get("HTTP_PORT"))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=DEBUG,
        use_reloader=False  # with reloader, caused main to be called twice
    )
