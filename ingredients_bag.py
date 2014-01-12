import json
import requests
import logging
import pickle
import nltk

## tornado pretty logging for non-tornado scripts
import tornado.options
import tornado.log
tornado.options.options.logging = "debug"
tornado.log.enable_pretty_logging()

FILENAME = "ingredients_tokenized.pickle"

class Ingredients_Bag(object):

    def load_from_yummly_dump(self):
        ingred_dict = json.load(file("ingredient.json"))
        ingred_arr = [a.values()[0].encode('ascii', 'ignore') \
                      for a in ingred_dict["en-US"]]
        self.bag = [nltk.word_tokenize(a.lower()) for a in ingred_arr]

    def write_to_file(self):
        pickle.dump(self.bag, open(FILENAME, "wb"))

    def load_from_file(self):
        self.bag = pickle.load(open(FILENAME, "rb"))

    def get_matches(self, text):
        sentences = nltk.sent_tokenize(text)
        tags = []
        for s in sentences:
            ngrams = nltk.trigrams(nltk.word_tokenize(s.lower()))
            ngrams += nltk.bigrams(nltk.word_tokenize(s.lower()))
            ngrams += (nltk.word_tokenize(s.lower()))
            for t in ngrams:
                tt = ([t] if isinstance(t, str) else list(t))
                if tt in self.bag:
                    tags.append((t, 'IGRD'))
        return tags
