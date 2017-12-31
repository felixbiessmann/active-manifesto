import sqlite3
import os

DB_FILENAME = os.environ.get('DB_PATH')
print('Using database', DB_FILENAME)


def insert_into(text_ids, labels, annotation_source):
    """
    inserts the texts and labels.

    :param text_ids: iterable of int, ids of the texts.
    :param labels: iterable of int.
    :param annotation_source: string, manifesto or user.
    """
    conn = sqlite3.connect(DB_FILENAME)
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


def get_texts_only(n_texts):
    """
    return the text data only without labels, at most `n_texts`.

    :param n_texts: how many texts to retrieve, integer.
    :return: majority voted labels per text: [
        {'text_id': 1, 'statement': 'some statement', 'label': 'left'},
        ...
    ]
    """
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    results = [
        {
            'text_id': text_id,
            'statement': statement
        } for text_id, statement in c.execute(
            """
            SELECT
                t.id text_id,
                t.statement
            FROM texts t
            LIMIT ?""",
            (n_texts, )
        )
    ]
    return results


def get_texts_with_labels(n_texts, label_strategy='duplicate'):
    """
    return at most `n_texts` from the underlying storage

    :param n_texts: how many texts to retrieve, integer.
    :param label_strategy:  one of ['duplicate', 'majority']
                            duplicate - repeat each training text as often as there are labels for it.
                            majority -  each training text is returned exactly once, with majority voted label,
                                        in break even situations a random label is chosen
    :return: majority voted labels per text: [
        {'text_id': 1, 'statement': 'some statement', 'label': 'left'},
        ...
    ]
    """
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    if label_strategy == 'majority':
        results = [
            {
                'text_id': text_id,
                'statement': statement,
                'labels': labels
            } for text_id, statement, labels in c.execute(
                """
                SELECT
                    t.id text_id,
                    t.statement,
                    GROUP_CONCAT(l.label) labels
                FROM texts t
                INNER JOIN labels l ON l.texts_id = t.id
                GROUP BY t.id, t.statement
                LIMIT ?""",
                (n_texts, )
            )
        ]
        results = list(
            map(
                lambda r: {'text_id': r['text_id'], 'statement': r['statement'], 'label': max(set(r['labels'].split(',')), key=r['labels'].count)},
                results
            )
        )
        return results
    else:
        results = [
            {
                'text_id': text_id,
                'statement': statement,
                'label': label
            } for text_id, statement, label in c.execute(
                """
                SELECT
                    t.id text_id,
                    t.statement,
                    l.label label
                FROM texts t
                INNER JOIN labels l ON l.texts_id = t.id
                LIMIT ?""",
                (n_texts, )
            )
        ]
        return results
