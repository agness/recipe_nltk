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

def __get_meta_tag(soup, keys):
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

def scrape(url):
    result = {}
    try:
        markup = requests.get(url).text
        soup = BeautifulSoup(markup)
        # get meta tags before we strip anything
        result["source"] = __get_meta_tag(soup, ["site_name"])
        result["title"] = __get_meta_tag(soup, ["title"])
        result["url"] = __get_meta_tag(soup, ["url"])
        result["description"] = __get_meta_tag(soup, ["description"])
        # Strip script tags and comments
        [s.extract() for s in soup(['script','style','comments','header','footer'])]
        # Get only text trimmed of whitespace and punctuation
        result["body"] = list(soup.stripped_strings) # returns generator
    except Exception as e:
        raise # debug for now!
        logging.error("({0}): {1}, {2}".format(type(e), e.args, e))
    return result

def test():
    URLS = [
        "http://www.marthastewart.com/874410/rack-lamb-mustard-sauce",
        "http://www.epicurious.com/recipes/food/views/Toffee-Crunch-Caramel-Cheesecake-231417",
        "http://www.bonappetit.com/recipe/googles-braised-chicken-and-kale",
        "http://www.foodnetwork.com/recipes/ellie-krieger/pasta-puttanesca-recipe/index.html",
        "http://www.chow.com/recipes/10652-pasta-carbonara-with-peas",
        "http://www.simplyrecipes.com/recipes/lemon_pesto_turkey_pasta/",
        "http://www.realsimple.com/food-recipes/browse-all-recipes/pasta-broccoli-rabe-sausage-10000001094505/",
        "http://southernfood.about.com/od/copycatrecipes/r/blcc20.htm",
        "http://www.foodandwine.com/recipes/pasta-with-sausage-basil-and-mustard",
        "http://thepioneerwoman.com/cooking/2010/07/pasta-with-pancetta-and-leeks/",
        "http://thaifood.about.com/od/thaisnacks/r/greenmangosalad.htm",
        "http://www.101cookbooks.com/archives/a-good-winter-salad-recipe.html",
        "http://www.kraftrecipes.com/recipes/watergate-salad-53771.aspx",
        "http://smittenkitchen.com/blog/2013/12/gingerbread-snacking-cake/",
        "http://food52.com/recipes/25815-sherlock-watson",
        "http://leitesculinaria.com/83239/recipes-sidecar-cocktail.html",
        "http://liquor.com/recipes/sidecar/"
    ]

    for url in URLS:
        print "-" * 10
        print json.dumps(scrape(url), indent=2)

def main():
    test()

if __name__ == "__main__":
    main()
