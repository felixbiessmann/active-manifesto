# coding: utf-8

from flask import Flask
from flask import render_template, redirect, url_for, request
import json


app = Flask(__name__)
app.debug = True
app.secret_key = 'development'


@app.route('/user_labels', methods=['POST'])
def user_labels():
    # fixme: check bytes encoding here
    # todo: forward to model http facade and receive bias estimation
    labels = json.loads(request.data.decode('utf-8'))
    print(labels)
    for entry in labels['data']:
        print(entry)
    return '{}'

@app.route('/get_samples')
def get_samples():
    # todo: call model http facade here
    response = {
        'samples': [
            {'text': 'abc'},
            {'text': 'abcd'}
        ]
    }
    return json.dumps(response)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
