import bs4 as bs
import urllib.request
import re


class NewsEngine:

    def __init__(self):

        self.source = urllib.request.urlopen('https://edition.cnn.com/articles/').read()
        self.base_url = "https://edition.cnn.com"
        self.article_titles = []
        self.article_links = []

    def get_latest_articles(self):

        soup = bs.BeautifulSoup(self.source, "lxml")

        # Fetch through all div
        for div in soup.find_all('div'):

            # Get div class
            cls = div.get('class')

            # Check if class is a list, somtimes it have None type
            if isinstance(cls, list):

                # Get the first item of class list
                if cls[0] == "cd__content":

                    # Get article title
                    a_child = div.findChildren("a", recursive=True)[0]
                    self.article_links.append(self.base_url + str(a_child.get('href')))

                    # Get article link
                    span_child = div.findChildren("span", recursive=True)[0]
                    self.article_titles.append(span_child.get_text())

        return self.article_titles, self.article_links

    def get_article(self, article_number):

        articles, links = self.get_latest_articles()
        source = urllib.request.urlopen(links[article_number - 1]).read()
        soup = bs.BeautifulSoup(source, "lxml")

        article_text = ""

        # Fetch through all div

        for div in soup.find_all("div", {"class": "el__leafmedia el__leafmedia--sourced-paragraph"}):

            article_text = article_text + str(div.get_text()) + " "

        for div in soup.find_all("div", {"class": "zn-body__paragraph"}):

            article_text = article_text + str(div.get_text()) + " "

        return article_text