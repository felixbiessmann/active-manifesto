# -*- coding: utf-8 -*-
import os
import manifesto_data

from flask import Flask, jsonify, request
import sqlite3
import urllib.parse

app = Flask(__name__)

DEBUG = True  # os.environ.get('DEBUG') is not None
VERSION = 0.1


@app.route("/", methods=['POST'])
def index():
    return jsonify({})


@app.route("/texts", methods=['GET'])
def texts():
    n_texts = urllib.parse.quote(request.args.get('n'))
    text_data = {'data': get_texts(n_texts)}
    return jsonify(text_data)


def get_texts(n_texts):
    conn = sqlite3.connect('/tmp/db/test.db')
    c = conn.cursor()
    return [
        {
            'statement': statement,
            'label': label,
            'source': source
        } for statement, label, source, _, _ in c.execute(
        """
        SELECT
            t.statement,
            l.label,
            l.source,
            t.created_at,
            l.created_at
        FROM texts t
        INNER JOIN labels l ON l.texts_id = t.id
        LIMIT ?""",
        (n_texts, )
        )
    ]


def create_manifesto_storage(texts, labels):
    conn = sqlite3.connect('/tmp/db/test.db')
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

    pks = []
    for text in texts:
        c.execute(
            "INSERT INTO texts(statement) VALUES (?)",
            (text, )
        )
        pks.append(c.lastrowid)

    for text_id, label in zip(pks, labels):
        c.execute(
            """
            INSERT INTO labels (texts_id, label, source) VALUES
            (?, ?, ?)""",
            (text_id, label, 'manifesto_project')
        )

    conn.commit()

    for statement, label, source, text_created, label_created in c.execute(
            """
            SELECT
                t.statement,
                l.label,
                l.source,
                t.created_at,
                l.created_at
            FROM texts t
            INNER JOIN labels l ON l.texts_id = t.id
            LIMIT 10"""
    ):
        print(statement, label, source)

    print(
        'inserted',
        c.execute("SELECT count(1) FROM texts").fetchone(),
        'texts with',
        c.execute("SELECT count(1) FROM labels").fetchone(),
        'labels'
    )

    conn.close()


if __name__ == "__main__":
    api_key = os.environ.get("WZB_API_KEY")

    loader = manifesto_data.ManifestoDataLoader(api_key)
    texts, codes = loader.get_manifesto_texts()

    create_manifesto_storage(texts, codes)

    port = os.environ.get("HTTP_PORT")
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
