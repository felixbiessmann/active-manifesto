# coding: utf-8

import json
import os
import requests
import urllib

from flask import Flask, jsonify
from flask import redirect, render_template, request

from newsreader import NewsReader
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
DEBUG=True
MANIFESTO_MODEL_HTTP_PORT = os.environ.get('MANIFESTO_MODEL_HTTP_PORT')
HTTP_PORT = int(os.environ.get('HTTP_PORT'))

MANIFESTO_URL = 'http://manifesto_model:{}/predict'.format(MANIFESTO_MODEL_HTTP_PORT)

reader = NewsReader()
scheduler = BackgroundScheduler()
scheduler.add_job(reader.fetch_news, 'interval', minutes=60)
scheduler.start()

@app.route('/get_topics')
def get_topics():
    """
    use this endpoint to get n news topics

    e.g.: http://localhost:5000/get_topics?n=5
    :return:
    """
    n_topics = int(request.args.get('n', default=5))

    topics = []
    for t,v in reader.topics[:n_topics]:
         topics.append({"topic":t, "variance_explained":v})

    return json.dumps(topics)

@app.route('/get_news')
def get_news():
    """
    use this endpoint to get left or right biased news.

    e.g.: http://localhost:5000/get_news?bias=left
    :return:
    """
    articles = reader.articles
    response = []
    bias = request.args.get('bias', default="")
    if bias.lower().strip() in ['left', 'right', 'neutral']:
        for article in articles:
            # check if article was assigned label [bias]
            if sorted(article['label'].items(),key=lambda x:x[1])[-1][0]==bias:
                response.append(article)
    else:
        response = articles
    return json.dumps(response)

if __name__ == '__main__':

    import time
    time.sleep(10)
    reader.fetch_news()

    app.run(host="0.0.0.0", port=HTTP_PORT, debug=DEBUG)
