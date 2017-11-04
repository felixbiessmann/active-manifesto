# coding: utf-8

from flask import Flask
from flask import render_template, redirect, url_for
import json


app = Flask(__name__)
app.debug = True
app.secret_key = 'development'


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
