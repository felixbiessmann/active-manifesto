# -*- coding: utf-8 -*-
import urllib.request

import readability
from bs4 import BeautifulSoup


class NewsReader(object):
    def __init__(self, sources=['spiegel', 'faz', 'welt', 'zeit']):
        """
        :param sources: a list of strings for each newspaper for which a crawl is
        implemented
        """
        self.sources = sources

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

            if source is 'spiegel':
                # fetching articles from sueddeutsche.de/politik
                url = 'http://www.spiegel.de/politik'
                site = BeautifulSoup(urllib.request.urlopen(url).read())
                titles = site.findAll("div", {"class": "teaser"})
                urls = ['http://www.spiegel.de' + a.findNext('a')['href'] for a in titles]

            if source is 'faz':
                # fetching articles from sueddeutsche.de/politik
                url = 'http://www.faz.net/aktuell/politik'
                site = BeautifulSoup(urllib.request.urlopen(url).read())
                titles = site.findAll("a", {"class": "TeaserHeadLink"})
                urls = ['http://www.faz.net' + a['href'] for a in titles]

            if source is 'welt':
                # fetching articles from sueddeutsche.de/politik
                url = 'http://www.welt.de/politik'
                site = BeautifulSoup(urllib.request.urlopen(url).read())
                titles = site.findAll("a", {"class": "as_teaser-kicker"})
                urls = [a['href'] for a in titles]

            if source is 'sz-without-readability':
                # fetching articles from sueddeutsche.de/politik
                url = 'http://www.sueddeutsche.de/politik'
                site = BeautifulSoup(urllib.request.urlopen(url).read())
                titles = site.findAll("div", {"class": "teaser"})
                urls = [a.findNext('a')['href'] for a in titles]

            if source is 'zeit':
                # fetching articles from zeit.de/politik
                url = 'http://www.zeit.de/politik'
                site = BeautifulSoup(urllib.request.urlopen(url).read())
                urls = [a['href'] for a in site.findAll("a", {"class": "teaser-small__combined-link"})]

            print("Found %d articles on %s" % (len(urls), url))

            for url in urls:
                try:
                    title, text = NewsReader.fetch_url(url)
                    articles.append((title, text))
                except RuntimeError:
                    print('Could not get text from %s' % url)
                    pass

        return articles


if __name__ == "__main__":
    reader = NewsReader(sources=['spiegel'])
    for news_item in reader.get_news():
        print(news_item)
