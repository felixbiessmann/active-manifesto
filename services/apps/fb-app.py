import requests, os, binascii
from flask import Flask, redirect, url_for, session, request, jsonify
from flask_oauthlib.client import OAuth, OAuthException

FACEBOOK_APP_ID = os.environ['FACEBOOK_APP_ID']
FACEBOOK_APP_SECRET = os.environ['FACEBOOK_APP_SECRET']

app = Flask(__name__)
app.debug = True
app.secret_key = binascii.hexlify(os.urandom(24))
oauth = OAuth(app)

facebook = oauth.remote_app(
    'facebook',
    consumer_key=FACEBOOK_APP_ID,
    consumer_secret=FACEBOOK_APP_SECRET,
    request_token_params={'scope': 'email'},
    base_url='https://graph.facebook.com',
    request_token_url=None,
    access_token_url='/oauth/access_token',
    access_token_method='GET',
    authorize_url='https://www.facebook.com/dialog/oauth'
)


def get_fb_likes(user_name, access_token, max_likes=1000):
    '''

    Slow retrieval of user likes

    '''
    endpoint = 'https://graph.facebook.com/v2.10/{}/likes?access_token={}'
    url = endpoint.format(user_name,access_token)
    likes = requests.get(url).json()
    all_likes = likes['data']
    while "next" in likes['paging'] and len(all_likes) < max_likes:
        likes = requests.get(likes['paging']['next']).json()
        all_likes += likes['data']
    return all_likes


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/login')
def login():
    callback = url_for(
        'facebook_authorized',
        next=request.args.get('next') or request.referrer or None,
        _external=True
    )
    return facebook.authorize(callback=callback)


@app.route('/login/authorized')
def facebook_authorized():
    resp = facebook.authorized_response()
    if resp is None:
        return 'Access denied: reason=%s error=%s' % (
            request.args['error_reason'],
            request.args['error_description']
        )
    if isinstance(resp, OAuthException):
        return 'Access denied: %s' % resp.message

    session['oauth_token'] = (resp['access_token'], '')

    likes = get_fb_likes("me",resp['access_token'])

    return ", ".join([like['name'] for like in likes])

@facebook.tokengetter
def get_facebook_oauth_token():
    return session.get('oauth_token')

if __name__ == '__main__':
    app.run()
