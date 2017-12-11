# coding: utf-8

import json
import os
import requests

from flask import Flask
from flask import render_template, request

app = Flask(__name__)
app.debug = True
app.secret_key = 'development'

PERSISTENCE_HTTP_PORT = os.environ.get('PERSISTENCE_HTTP_PORT')
HTTP_PORT = int(os.environ.get('HTTP_PORT'))


@app.route('/user_labels', methods=['POST'])
def user_labels():
    """
    receives the texts and labels from the UI.
    """
    labels = json.loads(request.get_data(as_text=True))
    # registered hostname of service in docker-compose network
    url = 'http://persistence:{}/texts_and_labels'.format(PERSISTENCE_HTTP_PORT)
    print(labels)
    r = requests.post(
        url=url,
        json=labels
    )
    print(r.status_code)
    print(r.json())
    return '{}'


@app.route('/get_samples')
def get_samples(mock=True):
    n_texts = request.args.get('n')
    if not mock:
        r = requests.get(
            'http://persistence:{}/texts?n={}'.format(
                PERSISTENCE_HTTP_PORT,
                n_texts
            )
        )
        return json.dumps(r.json())
    else:
        import time
        t = str(int(time.time()))
        return json.dumps(
            {
                'data': [
                    {'text_id': 1, 'statement': 'A ' + t},
                    {'text_id': 2, 'statement': 'B ' + t},
                    {'text_id': 3, 'statement': 'C ' + t},
                ]
            }
        )


@app.route('/swipe')
def swipe():
    return render_template(
        'swipe.html',
        persistence_host='http://0.0.0.0:'+str(PERSISTENCE_HTTP_PORT)
    )


@app.route('/')
def index():
    return render_template(
        'index.html',
        persistence_host='http://0.0.0.0:'+str(PERSISTENCE_HTTP_PORT)
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=HTTP_PORT)
