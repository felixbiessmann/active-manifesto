import sqlite3
import os
from manifesto_data import ManifestoDataLoader


DB_FILENAME = './db/active-manifesto.db'


def insert_into(database_filename, texts, labels, annotation_source):
    """
    inserts the texts and labels.

    :param database_filename: name of the sqlite3 database.
    :param texts: iterable of string.
    :param labels: iterable of int.
    :param annotation_source: string, manifesto or user.
    """
    conn = sqlite3.connect(database_filename)
    c = conn.cursor()

    pks = []
    for text in texts:
        c.execute(
            "INSERT INTO texts(statement) VALUES (?)",
            (text,)
        )
        pks.append(c.lastrowid)

    for text_id, label in zip(pks, labels):
        c.execute(
            """
            INSERT INTO labels (texts_id, label, source) VALUES
            (?, ?, ?)""",
            (text_id, label, annotation_source)
        )

    conn.commit()


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


if __name__ == "__main__":
    api_key = os.environ.get('WZB_API_KEY')
    loader = ManifestoDataLoader(api_key)
    text, code = loader.get_manifesto_texts()
    create_manifesto_storage(text, code)

