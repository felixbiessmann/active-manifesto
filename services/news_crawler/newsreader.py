# -*- coding: utf-8 -*-
import urllib.request

import os

import apscheduler.schedulers.background
import readability
from bs4 import BeautifulSoup
from pymongo import MongoClient

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

STOPWORDS = [x.strip() for x in open('stopwords.txt').readlines()[6:]]

class NewsReader(object):
    def __init__(self, sources=['nachrichtenleicht', 'spiegel', 'faz', 'welt', 'zeit']):
        """
        :param sources: a list of strings for each newspaper for which a crawl is
        implemented
        """
        self.sources = sources

    @staticmethod
    def get_topics(texts, n_topics=5):
        """
        Runs some topic modelling on text database
        INPUT:
        texts   iterable of texts
        n_topics    number of topics
        RETURNS
        assignments topic assignments for texts
        topics  list of (topic-string, variance_explained)
        """
        vect = CountVectorizer(stop_words=STOPWORDS).fit(texts)
        idx2word = {idx:w for w,idx in vect.vocabulary_.items()}
        X = vect.transform(texts)
        pca = PCA().fit(X.toarray())
        sim = pca.components_[:n_topics,:].dot(X.T.toarray())
        topics = [texts[c_idx] for c_idx in sim.argmax(axis=1).flatten()]
        assignments = pca.transform(X.toarray()).argmax(axis=1)

        return assignments, zip(topics,pca.explained_variance_ratio_)

    @staticmethod
    def fetch_url(url):
        """
        get url with readability
        """
        html = urllib.request.urlopen(url).read()
        readable_html = readability.Document(html)
        readable_article = readable_html.summary()
        title = readable_html.short_title()
        text = BeautifulSoup(readable_article).get_text()
        return title, text

    def get_news(self):
        """
        Collects all news articles from political ressort of major German newspapers
        and returns a list of tuples of (title, article_text).
        """

        articles = []

        # the classifier for prediction of political attributes

        for source in self.sources:

            url = None

            if source is 'nachrichtenleicht':
                url = 'http://www.nachrichtenleicht.de/nachrichten.2005.de.html'
                site = BeautifulSoup(urllib.request.urlopen(url).read(), "lxml")
                titles = site.findAll("p", {"class": "dra-lsp-element-headline"})
                urls = ['http://www.nachrichtenleicht.de/' + a.findNext('a')['href'] for a in titles]

            if source is 'spiegel':
                # fetching articles from sueddeutsche.de/politik
                url = 'http://www.spiegel.de/politik'
                site = BeautifulSoup(urllib.request.urlopen(url).read(), "lxml")
                titles = site.findAll("div", {"class": "teaser"})
                urls = ['http://www.spiegel.de' + a.findNext('a')['href'] for a in titles]

            if source is 'faz':
                # fetching articles from sueddeutsche.de/politik
                url = 'http://www.faz.net/aktuell/politik'
                site = BeautifulSoup(urllib.request.urlopen(url).read(), "lxml")
                titles = site.findAll("a", {"class": "tsr-Base_ContentLink"})
                urls = ['http://www.faz.net' + a['href'] for a in titles if  a.has_attr('href')]

            if source is 'welt':
                # fetching articles from sueddeutsche.de/politik
                url = 'http://www.welt.de/politik'
                site = BeautifulSoup(urllib.request.urlopen(url).read(), "lxml")
                titles = site.findAll("a", {"class": "o-link o-teaser__link "})
                urls = ['http://www.welt.de' + a['href'] for a in titles]

            if source is 'sz':
                # fetching articles from sueddeutsche.de/politik
                url = 'http://www.sueddeutsche.de/politik'
                site = BeautifulSoup(urllib.request.urlopen(url).read(), "lxml")
                titles = site.findAll("a", {"class": "entry-title"})
                urls = [a['href'] for a in titles if  a.has_attr('href')]

            if source is 'zeit':
                # fetching articles from zeit.de/politik
                url = 'http://www.zeit.de/politik'
                site = BeautifulSoup(urllib.request.urlopen(url).read(), "lxml")
                urls = [a['href'] for a in site.findAll("a", {"class": "teaser-small__combined-link"})]

            print("Found %d articles on %s" % (len(urls), url))

            for url in urls:
                try:
                    title, text = NewsReader.fetch_url(url)
                    articles.append(
                        {   'title': title,
                            'text': text,
                            'source': source,
                            'url': url
                            })
                except:
                    print('Could not get text from %s' % url)
                    pass

            topic_assignments, topics = self.get_topics([x['title'] for x in articles])
            for article, topic_assignment in zip(articles,topic_assignments):
                article['topic'] = topic_assignment

        return articles, topics


def fetch_news():
    reader = NewsReader(sources=['spiegel'])

    persistence = MongoClient()

    db = persistence['test-database']
    col = db['articles']
    col.delete_many({})
    col.insert_many(reader.get_news())


if __name__ == "__main__":
    os.system("mongod&")  # fixme: quick hack to make mongod run

    # the scheduler is the only piece of code this container is running
    # so using blocking scheduler is ok.
    scheduler = apscheduler.schedulers.background.BlockingScheduler()
    # fixme: increase the interval for production
    scheduler.add_job(fetch_news, 'interval', minutes=1)
    scheduler.start()
