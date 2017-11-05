# coding: utf-8

from flask import Flask
from flask import render_template, redirect, url_for, request
import json


app = Flask(__name__)
app.debug = True
app.secret_key = 'development'


@app.route('/user_labels', methods=['POST'])
def user_labels():
    # todo: bias estimation of this request and return estimate in response
    labels = json.loads(request.data.decode('utf-8'))
    print(labels)
    for entry in labels['data']:
        print(entry)
    return '{}'


@app.route('/get_samples')
def get_samples():
    # todo: actually use model to return samples
    response = {
        'samples': [
            {'text': 'abc'},
            {'text': 'abcd'}
        ]
    }
    return json.dumps(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
