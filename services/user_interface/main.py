# coding: utf-8

import json
import os
import requests
import random

from flask import Flask, jsonify
from flask import redirect, render_template
from flask import g, session, request, url_for, flash
from flask_oauthlib.client import OAuth


app = Flask(__name__)
app.debug = True
app.secret_key = 'development'

MANIFESTO_MODEL_HTTP_PORT = os.environ.get('MANIFESTO_MODEL_HTTP_PORT')
NEWS_CRAWLER_HTTP_PORT = os.environ.get('NEWS_CRAWLER_HTTP_PORT')
HTTP_PORT = int(os.environ.get('HTTP_PORT'))

# twitter oauth and endpoint configuration
oauth = OAuth(app)
twitter = oauth.remote_app(
    'Active Manifesto twitter-integration',
    consumer_key='xBeXxg9lyElUgwZT6AZ0A',
    consumer_secret='aawnSpNTOVuDCjx7HMh6uSXetjNN8zWLpZwCEU4LBrk',
    base_url='https://api.twitter.com/1.1/',
    request_token_url='https://api.twitter.com/oauth/request_token',
    access_token_url='https://api.twitter.com/oauth/access_token',
    authorize_url='https://api.twitter.com/oauth/authorize'
)


@twitter.tokengetter
def get_twitter_token():
    if 'twitter_oauth' in session:
        resp = session['twitter_oauth']
        return resp['oauth_token'], resp['oauth_token_secret']


@app.before_request
def before_request():
    g.user = None
    if 'twitter_oauth' in session:
        g.user = session['twitter_oauth']


@app.route('/twitter')
def twitter_index():

    def add_political_direction_to(tweets):
        url = 'http://manifesto_model:{}/predict'.format(MANIFESTO_MODEL_HTTP_PORT)
        for tweet in tweets:
            tweet['labels'] = requests.post(url=url, json={'text': tweet['text']}).json()['prediction']
        pass

    tweets = None
    if g.user is not None:
        resp = twitter.request('statuses/home_timeline.json?count=10')  # todo: make configurable
        if resp.status == 200:
            tweets = resp.data
            add_political_direction_to(tweets)
        else:
            print(resp.raw_data)
            flash('Unable to load tweets from Twitter.')
    return render_template('twitter.html', tweets=tweets)


@app.route('/search', methods=['GET'])
def search():
    """
    use this endpoint to get most popular query tweets with some additional stats.

    e.g.: http://localhost:5000/search?q=tagesschau
    :return:
    """
    q = urllib.parse.quote(request.args.get('q'))
    # https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets.html
    a = twitter.get('search/tweets.json?q={}&locale=en-US&result_type=popular&count=100'.format(q))
    tweets = [
        "{user_name}({n_followers}) - {tweet} - {n_retweets}".format(
            user_name=status['user']['name'],
            n_followers=status['user']['followers_count'],
            n_retweets=status['retweet_count'],
            tweet=status['text']
        ) for status in a.data['statuses']
    ]
    return '<br><br>'.join(tweets)


@app.route('/twitter/login')
def twitter_login():
    callback_url = url_for('twitter_oauthorized', next=request.args.get('next'))
    return twitter.authorize(callback=callback_url or request.referrer or None)


@app.route('/twitter/logout')
def twitter_logout():
    session.pop('twitter_oauth', None)
    return redirect(url_for('twitter_index'))


@app.route('/twitter/oauthorized')
def twitter_oauthorized():
    resp = twitter.authorized_response()
    if resp is None:
        flash('You denied the request to sign in.')
    else:
        session['twitter_oauth'] = resp
    return redirect(url_for('twitter_index'))


# labeling and prediction routes
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

@app.route('/get_topics_and_news')
def get_topics_and_news():
    """
    use this endpoint to get news topics.

    e.g.: http://localhost:5000/get_topics_and_news?n=5&bias=left
    :return:
    """
    n_topics = int(request.args.get('n', default=5))
    bias = request.args.get('bias', default="")
    topics = requests.get(
        'http://news_crawler:{}/get_topics?n={}'.format(
            NEWS_CRAWLER_HTTP_PORT,
            n_topics
        )
    ).json()

    news = requests.get(
        'http://news_crawler:{}/get_news?bias={}'.format(
            NEWS_CRAWLER_HTTP_PORT,
            request.args.get('bias', default="")
        )
    ).json()

    # for diversity of articles and reduction of visual noise, choose 1 article
    for topic_idx, topic in enumerate(topics):
        topic['articles'] = [ article for article in news
                                if article['topic'] == topic_idx ]
    topics = [random.choice(topic['articles']) for topic in topics if len(topic['articles']) > 0]
    return json.dumps(topics)

@app.route('/get_topics')
def get_topics():
    """
    use this endpoint to get news topics.

    e.g.: http://localhost:5000/get_topics?n=5
    :return:
    """
    n_topics = int(request.args.get('n', default=5))
    r = requests.get(
        'http://news_crawler:{}/get_topics?n={}'.format(
            NEWS_CRAWLER_HTTP_PORT,
            n_topics
        )
    )
    return json.dumps(r.json())

@app.route('/get_news')
def get_news(n=5):
    r = requests.get(
        'http://news_crawler:{}/get_news?bias={}'.format(
            NEWS_CRAWLER_HTTP_PORT,
            request.args.get('bias', default="")
        )
    )
    n = int(request.args.get('n', default=n))
    return json.dumps(r.json()[:n])

@app.route('/get_samples')
def get_samples():
    n_texts = request.args.get('n')
    r = requests.get(
        'http://manifesto_model:{}/prioritized_texts_with_label?n={}'.format(
            MANIFESTO_MODEL_HTTP_PORT,
            n_texts
        )
    )
    return json.dumps(r.json())


@app.route('/predict', methods=['POST'])
def predict():
    req = json.loads(request.get_data(as_text=True))
    print('ui req', req)
    # registered hostname of service in docker-compose network
    url = 'http://manifesto_model:{}/predict'.format(MANIFESTO_MODEL_HTTP_PORT)
    r = requests.post(url=url, json=req)
    print(r.status_code)
    return jsonify(r.json())


@app.route('/debug/uncertainties')
def debug_uncertainties():
    # registered hostname of service in docker-compose network
    url = 'http://manifesto_model:{}/debug/uncertainties'.format(MANIFESTO_MODEL_HTTP_PORT)
    r = requests.get(url=url)
    return jsonify(r.json())


@app.route('/viz')
def viz():
    return render_template('viz.html')


@app.route('/rightornot')
def rightornot():
    return render_template('rightornot.html')


@app.route('/swipe')
def swipe():
    return render_template('swipe.html')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=HTTP_PORT)
