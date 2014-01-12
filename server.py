#!/usr/bin/env python
# agnes is responsible.

import json
import requests
import logging

import os
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.options
from tornado.options import options

from nltk_classifier import Recipe_NLTK_Classifier
from pattern_classifier import Recipe_Pattern_Classifier
from scrape import scrape

tornado.options.define("port", default=8888, help="port", type=int)

def load_classifier():
    global classifier
    classifier = Recipe_Pattern_Classifier()
    classifier.load_from_file()
    logging.debug("Using classifier %s" % type(classifier))

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
                    self.write("<i>%s, %s</i><br><br>" % classifier.classify(line))
        elif url:
            page_data = scrape(url)
            self.write("<p><b>source:</b> "+page_data["source"]+"</p>")
            self.write("<p><b>title:</b> "+page_data["title"]+"</p>")
            self.write("<p><b>url:</b> "+page_data["url"]+"</p>")
            self.write("<p><b>description:</b> "+page_data["description"]+"</p>")
            self.write("<hr>")
            for line in page_data["body"]:
                if len(line) > 5: # TODO drop any line < 2 words; NER for time
                    score = classifier.classify(line)
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
