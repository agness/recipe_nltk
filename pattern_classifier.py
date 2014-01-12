#!/usr/bin/env python
# agnes is responsible.

import os
import glob
import pickle
from datetime import datetime

import pattern.vector # import KNN, count
import pattern.en
import collections

from ingredients_bag import Ingredients_Bag

import logging
## tornado pretty logging for non-tornado scripts
import tornado.options
import tornado.log
tornado.options.options.logging = "debug"
tornado.log.enable_pretty_logging()

"""
Elementary recipe NLP classifier powered by pattern.
http://www.clips.ua.ac.be/pattern

Expects data to be labeled in format:
   { label <string> : text <string> }
   OR
   { label <string> : text <[]string> }

Example recipe data format:
   {
       "name":        string
       "time":        string
       "yield":       string
       "ingredients": []string
       "preparation": []string
   }
"""
class Recipe_Pattern_Classifier(object):

    FILENAME = {
        "PREFIX": "dump_pattern_classifier-",
        "TSTAMP": "%Y%m%d-%H%M",
        "SUFFIX": ".pickle"
        }

    def __init__(self):
        self.ingredients_bag = Ingredients_Bag()
        self.ingredients_bag.load_from_file()
        pass

    def __get_latest_file(self):
        """
        Return the filename of the most recent classifier pickle file.
        """
        fn = self.FILENAME["PREFIX"] + "*" + self.FILENAME["SUFFIX"]
        archives = sorted(glob.glob(fn), key=os.path.getmtime)
        if len(archives) > 0:
            return archives[-1]  # last one on list is latest
        return None

    def __get_features(self, text):
        """
        Given a string, tokenize, tag, and return tag counts.

        Returns { feature: <int>, ... }
        """
        features = {}
        # -- POS tags
        pos_tags = [x[1] for x in pattern.en.tag(text)]
        tags_count = collections.Counter(pos_tags)
        # rename "." tag to "PERIOD" so avoid confusion in dot syntax keys
        features.update({"POS_"+tag:val for tag,val in tags_count.items()})
        # -- chunk tags
        sentences = pattern.en.parsetree(text)
        def get_chunk_tags(sentence):
            return [chunk.tag for chunk in sentence.chunks]
        chunks = [chunk for s in sentences for chunk in get_chunk_tags(s)]
        tags_count = collections.Counter(chunks)
        features.update({tag:val for tag,val in tags_count.items()})
        # -- our custom ingredients bag
        igrd_tags = [x[1] for x in self.ingredients_bag.get_matches(text)]
        tags_count = collections.Counter(igrd_tags)
        features.update({tag:val for tag,val in tags_count.items()})
         # TODO could also try # sent., # words, mean words per sent., but
        # these probably very different for nytimes v. (e.g.) Betty Crocker
        # return all features
        return features

    def __tag_record(self, k, v):
        """
        If given a string, returns tagged tuple, else returns None.
        """
        if isinstance(v, unicode) or isinstance(v, str):
            return pattern.vector.Document(self.__get_features(v), type=k)
        else:
            return None

    def __tag_data_set(self, s):
        """
        Given set of labeled recipe data, tag the entire set.
        """
        tagged = []
        for d in s:
            for k, v in d.items():
                if isinstance(v, list):
                    for vv in v:
                        tagged.append(self.__tag_record(k, vv))
                else:
                    tagged.append(self.__tag_record(k, v))
        return [t for t in tagged if t is not None]

    def train(self, d):
        """
        Given a labeled set, train our classifier.
        """
        t = self.__tag_data_set(d)
        self.classifier = pattern.vector.NB(train=t)
        logging.info("Training on %s records complete." % len(d))

    def accuracy(self, d):
        """
        Given a labeled test set, return the accuracy of our classifier.

        Returns float.
        """
        t = self.__tag_data_set(d)
        # Classifier.test() returns an (accuracy, precision, recall, F1-score)-tuple.
        return self.classifier.test(t)[0]

    def classify(self, s):
        """
        Classify a given string.

        Returns tuple:
            ( most-likely tag <string>, confidence <float> )
        """
        t = self.__get_features(s)
        logging.debug(t)
        import json # <<<<<<<<<<<<< DEBUGGGG
        r = self.classifier.classify(t, discrete=False)
        logging.debug("%s" % json.dumps(r))
        # get (key, probably) pair with max probability
        p = max(r.iterkeys(), key=(lambda k: r[k]))
        logging.debug("%s\n%s\n >>> %s\n" % \
            (s,json.dumps(t),json.dumps(p)))
        return (p, r[p])

    def save_to_file(self):
        """
        Save our classifier to file.
        """
        filename = self.FILENAME["PREFIX"] + self.FILENAME["TSTAMP"] + \
            self.FILENAME["SUFFIX"]
        self.classifier.save(datetime.now().strftime(filename))

    def load_from_file(self, infile=None):
        """
        Load our classifier from a file.
        """
        if not infile:
            infile = self.__get_latest_file()
        self.classifier = pattern.vector.NB.load(infile)
