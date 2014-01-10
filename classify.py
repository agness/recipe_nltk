#!/usr/bin/env python
# kudos to http://datadesk.latimes.com/posts/2013/12/natural-language-processing-in-the-kitchen/
# agnes made this.

import json
import requests
import logging

import nltk
import pickle
from nltk.classify import MaxentClassifier
from nltk.tag.simplify import simplify_wsj_tag
from bs4 import BeautifulSoup

## tornado pretty logging for non-tornado scripts
import tornado.options
import tornado.log
tornado.options.options.logging = "debug"
tornado.log.enable_pretty_logging()

API = "http://localhost:3000/api/recipes/" # <--- Adam API

def load_data():
    data = []
    for n in range(50,260): # <----------- choose how much data you want to load
        try:
            # Use Adam's nytcooking API, and keep only:
            #    name:        string
            #    time:        string
            #    yield:       string
            #    ingredients: array of strings
            #    preparation: array of strings
            d = requests.get(API+str(n)).json()
            d = d["recipe"]
            d.pop("byline", None)
            d.pop("badge", None)
            d.pop("image", None)
            d["ingred"] = []
            for v in d["ingredients"]:
                d["ingred"].append(v["quantity"]+v["name"])
            d["ingredients"] = d.pop("ingred")
            d["prep"] = []
            for v in d["preparation"]:
                d["prep"].append(v["description"])
            d["preparation"] = d.pop("prep")
            data.append(d)
        except Exception as e:
            logging.error("({0}): {1}, {2}".format(type(e), e.args, e))
    return data

def get_features(text):
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

def tagit(training_data):
    tagged_data = []
    for d in training_data:
        for k, v in d.items():
            if v: # e.g. {time: None}
                for i in v:
                    tagged_data.append((get_features(i), k))
    return tagged_data

def train():
    d = load_data()
    logging.info(d)
    t = tagit(d)
    logging.info(t)

    # Hackety-split our data 70% 30%, latter as test set
    split_ind = int(len(t) * 0.7)
    logging.info("split_index = "+str(split_ind))
    train_set, test_set = t[:split_ind], t[split_ind:]

    # Train a Maximum-Entropy Classifier
    global classifier
    classifier = MaxentClassifier.train(t)

    # Report accuracy
    logging.info(nltk.classify.accuracy(classifier, test_set))
    return classifier

def test(s):
    """ Use this if you wanted to try classifying a specific string """
    t = get_features(s)
    logging.info(t)
    c = classifier.classify(t)
    p = classifier.prob_classify(t).prob(c)
    logging.info(s+"\n"+json.dumps(t)+"\n >>> "+c+", "+str(p)+"\n")

def save_to_disk():
    """ Save our classifier object to disk (can load to save training time) """
    global classifier
    outfile = open('my_pickle.pickle', 'wb')
    pickle.dump(classifier, outfile)
    outfile.close()

#------------------------------------------

def main():
    train()
    test("Frittata With Turnips and Olives")
    test("6 servings.")
    test("1 medium onion, chopped")
    test("a bouquet garni made with a bay leaf, a Parmesan rind and a sprig each of sage, thyme and parsley")
    test("Meanwhile, wriggle a thin, sharp knife into each end of the meat, making a kind of pilot hole. Then use the handle of a long wooden spoon to force a hole all the way through the loin. Wriggle the spoon to make the hole as wide as you can. Stuff the apple and onion mixture into the roast from each end, all the way to the center. Sprinkle the roast with salt and pepper.")
    save_to_disk()

if __name__ == "__main__":
    main()
