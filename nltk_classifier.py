#!/usr/bin/env python
# kudos to http://datadesk.latimes.com/posts/2013/12/natural-language-processing-in-the-kitchen/
# agnes made this.

import logging
import glob
from datetime import datetime

import nltk
import pickle
from nltk.classify import MaxentClassifier
from nltk.tag.simplify import simplify_wsj_tag

## tornado pretty logging for non-tornado scripts
import tornado.options
import tornado.log
tornado.options.options.logging = "debug"
tornado.log.enable_pretty_logging()

"""
Elementary recipe NLP classifier powered by NLTK.

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
class Recipe_NLTK_Classifier(object):

    FILENAME = {
        "PREFIX": "nltk_classifier-",
        "TSTAMP": "%Y%m%d-%H%M",
        "SUFFIX": ".pickle"
        }

    def __init__(self):
        # dun have anything to do here yet
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
        Given a string, tokenize, tag, and return a normalized set of features.

        Returns ...
        """
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

    def __tag_record(self, k, v):
        """
        If given a string, returns tagged tuple, else returns None.
        """
        if isinstance(v, unicode) or isinstance(v, str):
            return (self.__get_features(v), k)
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
        self.classifier = MaxentClassifier.train(t)
        logging.info("Training on %s records complete." % len(d))

    def accuracy(self, d):
        """
        Given a labeled test set, return the accuracy of our classifier.

        Returns float.
        """
        t = self.__tag_data_set(d)
        return nltk.classify.accuracy(self.classifier, t)

    def classify(self, s):
        """
        Classify a given string.

        Returns tuple:
            ( most-likely tag <string>, confidence <float> )
        """
        t = self.__get_features(s)
        logging.debug(t)
        p = self.classifier.prob_classify(t)
        import json
        logging.debug("%s\n%s\n >>> %s, %s\n" % \
                     (s,json.dumps(t),p.max(),p.prob(p.max())))
        return (p.max(), p.prob(p.max()))

    def save_to_file(self):
        """
        Save our classifier to file.
        """
        filename = self.FILENAME["PREFIX"] + self.FILENAME["TSTAMP"] + \
            self.FILENAME["SUFFIX"]
        outfile = open(datetime.now().strftime(filename), 'wb')
        pickle.dump(self.classifier, outfile)
        outfile.close()

    def load_from_file(self, infile=None):
        """
        Load our classifier from a file.
        """
        if not infile:
            infile = self.__get_latest_file()
        classifier = pickle.load(open(infile, "rb"))
