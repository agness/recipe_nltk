#!/usr/bin/env python
# kudos to http://datadesk.latimes.com/posts/2013/12/natural-language-processing-in-the-kitchen/
# agnes made this.

import json
import requests
import logging

## tornado pretty logging for non-tornado scripts
import tornado.options
import tornado.log
tornado.options.options.logging = "debug"
tornado.log.enable_pretty_logging()

from nltk_classifier import Recipe_NLTK_Classifier
from pattern_classifier import Recipe_Pattern_Classifier

API = "http://localhost:3000/api/recipes/" # <--- Adam API

def load_data():
    data = []
    for n in range(50,100): # <----------- choose how much data you want to load
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

def main():
    d = load_data()
    # Hackety-split our data 80% 20%, and use latter as test set
    split_ind = int(len(d) * 0.8)
    logging.info("split_index = "+str(split_ind))
    train_set, test_set = d[:split_ind], d[split_ind:]

    # Train on training set
    classifier = Recipe_Pattern_Classifier()
    classifier.train(train_set)

    # Report accuracy on test set
    logging.info(classifier.accuracy(test_set))

    # Try some unlabeled stuff
    print classifier.classify("Frittata With Turnips and Olives")
    print classifier.classify("6 servings.")
    print classifier.classify("1 medium onion, chopped")
    print classifier.classify("a bouquet garni made with a bay leaf, a Parmesan rind and a sprig each of sage, thyme and parsley")
    print classifier.classify("Meanwhile, wriggle a thin, sharp knife into each end of the meat, making a kind of pilot hole. Then use the handle of a long wooden spoon to force a hole all the way through the loin. Wriggle the spoon to make the hole as wide as you can. Stuff the apple and onion mixture into the roast from each end, all the way to the center. Sprinkle the roast with salt and pepper.")

    # Should we save a copy?
    a = raw_input("Save classifier to file? (Y/n)")
    if a == "Y":
        classifier.save_to_file()

if __name__ == "__main__":
    main()
