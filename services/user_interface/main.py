# coding: utf-8

import json
import os
import requests

from flask import Flask
from flask import render_template, request

app = Flask(__name__)
app.debug = True
app.secret_key = 'development'

MANIFESTO_MODEL_HTTP_PORT = os.environ.get('MANIFESTO_MODEL_HTTP_PORT')
HTTP_PORT = int(os.environ.get('HTTP_PORT'))


@app.route('/user_labels', methods=['POST'])
def user_labels():
    """
    receives the texts and labels from the UI.
    """
    labels = json.loads(request.get_data(as_text=True))
    # registered hostname of service in docker-compose network
    url = 'http://manifesto_model:{}/texts_and_labels'.format(MANIFESTO_MODEL_HTTP_PORT)
    print(labels)
    r = requests.post(
        url=url,
        json=labels
    )
    print(r.status_code)
    print(r.json())
    return '{}'


@app.route('/get_samples')
def get_samples():
    n_texts = request.args.get('n')
    r = requests.get(
        'http://manifesto_model:{}/prioritized_texts?n={}'.format(
            MANIFESTO_MODEL_HTTP_PORT,
            n_texts
        )
    )
    return json.dumps(r.json())


@app.route('/swipe')
def swipe():
    return render_template(
        'swipe.html',
        # persistence_host='http://0.0.0.0:'+str(MANIFESTO_MODEL_HTTP_PORT)
    )


@app.route('/')
def index():
    return render_template(
        'index.html',
        # persistence_host='http://0.0.0.0:'+str(MANIFESTO_MODEL_HTTP_PORT)
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=HTTP_PORT)
