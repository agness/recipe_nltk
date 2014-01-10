#!/usr/bin/env python
# agnes is responsible.

import json
import requests
import logging

import nltk
import pickle
from nltk.classify import MaxentClassifier
from nltk.tag.simplify import simplify_wsj_tag
from bs4 import BeautifulSoup

import os
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.options
from tornado.options import options

tornado.options.define("port", default=8888, help="port", type=int)

# ------------------------------------------
# Classifier stuff

def load_classifier():
    global classifier
    classifier = pickle.load(open("classifier.pickle", "rb"))

def classify(s):
    """ this should be imported from a CLASSIFY module """
    t = get_features(s)
    #    l = classifier.classify(t)
    p = classifier.prob_classify(t)
    return (p.max(), p.prob(p.max()))

def get_features(text):
    """ this should be a CLASSIFY PRIVATE method """
    words = []
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = words + nltk.word_tokenize(sentence)
        pos = nltk.pos_tag(words)
        # TODO verify simplify_wsj_tag increases accuracy
        pos = [simplify_wsj_tag(tag) for word, tag in pos]
        words = [i.lower() for i in words]
        trigrams = nltk.trigrams(words)
        trigrams = ["%s/%s/%s" % (i[0], i[1], i[2]) for i in trigrams]
        features = words + pos + trigrams
        features = dict([(i, True) for i in features])
        return features

# ------------------------------------------
# Scrape external web pages

def scrape(url):
    result = {}
    try:
        markup = requests.get(url).text
        soup = BeautifulSoup(markup)
        # get meta tags before we strip anything
        result["source"] = get_meta_tag(soup, ["site_name"])
        result["title"] = get_meta_tag(soup, ["title"])
        result["url"] = get_meta_tag(soup, ["url"])
        result["description"] = get_meta_tag(soup, ["description"])
        # Strip script tags and comments
        [s.extract() for s in soup(['script','style','comments','header','footer'])]
        # Get only text trimmed of whitespace and punctuation
        result["body"] = list(soup.stripped_strings) # returns generator
    except Exception as e:
        raise # debug for now!
        logging.error("({0}): {1}, {2}".format(type(e), e.args, e))
    return result

def get_meta_tag(soup, keys):
    for k in keys:
        try:
            t = soup.find("meta", {"property":"og:"+k})
            return t["content"]
        except:
            pass
        try:
            t = soup.find("meta", {"property":k})
            return t["content"]
        except:
            pass
        try:
            t = soup.find("meta", {"name":k})
            return t["content"]
        except:
            pass
    return ""

# ------------------------------------------
# Web server endpoints

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("home.html")

    def post(self):
        blob = self.get_argument("blob", None)
        url = self.get_argument("url", None)
        self.write("<style>i{color:#aaa;}</style>")
        if blob:
            for line in blob.split("\n"):
                line = line.strip()
                if len(line) > 5:
                    self.write(line+"<br>")
                    self.write("<i>%s, %s</i><br><br>" % classify(line))
        elif url:
            page_data = scrape(url)
            self.write("<p><b>source:</b> "+page_data["source"]+"</p>")
            self.write("<p><b>title:</b> "+page_data["title"]+"</p>")
            self.write("<p><b>url:</b> "+page_data["url"]+"</p>")
            self.write("<p><b>description:</b> "+page_data["description"]+"</p>")
            self.write("<hr>")
            for line in page_data["body"]:
                if len(line) > 5:
                    score = classify(line)
                    if score[1] > 0.5:
                        self.write(line+"<br>")
                        self.write("<i>%s, %s</i><br><br>" % score)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            ( r"/", MainHandler ),
        ]
        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
        )
        tornado.web.Application.__init__(self, handlers, **settings)

# ------------------------------------------
# Main: load trained classifier from pickle and start a tornado instance

def main():
    load_classifier()
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    logging.info("Starting Tornado IOLoop on port %d...", options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
