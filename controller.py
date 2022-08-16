import asyncio
from aiohttp.client_exceptions import ClientConnectorError, ClientOSError
import aiohttp
from datetime import datetime
import requests
import re
from flask import jsonify
from tweet import Tweet
from tweets import Tweets
import time
from goose3 import Goose
from lxml.etree import ParserError
import logging
import model.model as model
from keras_preprocessing.sequence import pad_sequences
import numpy

TIMEOUT = 12
model_2, tokenizer = model.load_model()
#model_2 = model.load_model_bert()


################################################################################
################################################################################
############################Searchs#############################################
################################################################################
################################################################################
def search_counts(q):
    query = 'https://api.twitter.com/2/tweets/counts/recent?query={}'.format(q)
    return query


def search_trends():
    query = 'https://api.twitter.com/1.1/trends/place.json?id=23424977'
    return query


def search_conversation(id):
    query = 'https://api.twitter.com/2/tweets/search/recent?' \
            'query=conversation_id:{}&tweet.fields=in_reply_to_user_id,author_id,conversation_id'.format(id)
    return query


def search_tweets(q, links, images, retweet, reply, videos, media, urls, results, order, fields):
    SEARCH_RECENT_URL = 'https://api.twitter.com/2/tweets/search/recent?'

    constraints = ""

    if links == 0:
        constraints += " -has:links"
    elif links == 1:
        constraints += " has:links"
    else:
        constraints += ""

    if images == 0:
        constraints += " -has:images"
    elif images == 1:
        constraints += " has:images"
    else:
        constraints += ""

    if retweet == 0:
        constraints += " -is:retweet"
    elif retweet == 1:
        constraints += " is:retweet"
    else:
        constraints += ""

    if reply == 0:
        constraints += " -is:reply"
    elif reply == 1:
        constraints += " is:reply"
    else:
        constraints += ""

    if videos == 0:
        constraints += " -has:videos"
    elif videos == 1:
        constraints += " has:videos"
    else:
        constraints += ""

    if media == 0:
        constraints += " -has:media"
    elif media == 1:
        constraints += " has:media"
    else:
        constraints += ""

    if urls:
        constraints += ""
        for u in urls:
            constraints += ' -url:"{}"'.format(u)

    constraints += "&max_results={}".format(results)
    constraints += "&sort_order={}".format(order)
    constraints += "&tweet.fields={}".format(fields)

    query = "{}query={} {}".format(SEARCH_RECENT_URL, q, constraints)
    return query


def search_user(username):
    query = 'https://api.twitter.com/2/users/by/username/{}'.format(username)
    return query


def search_user_id(id):
    query = 'https://api.twitter.com/2/users/{}/tweets?max_results=100'.format(id)
    return query


################################################################################
################################################################################
##########################UTILS#################################################
################################################################################
################################################################################
# Without: 0, with: 1, does not matter: 2
def make_query(q, r, links=1, order="relevancy"):
    if links is not None:
        links = int(links)
    else:
        links = 1

    if order is None:
        order = "relevancy"

    images = 0
    retweet = 0
    reply = 0
    videos = 0
    media = 0
    results = r
    urls = ["https://twitter.com/", "https://youtu.be/", "https://instagram.com/"]
    fields = "referenced_tweets,public_metrics,entities,lang"

    return search_tweets(q, links, images, retweet, reply, videos, media, urls, results, order, fields)


def parse_date(date_string):
    return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ")


def parse_date_2(date_string):
    r = re.findall(r"\d+-\d+-\d+", date_string)
    return r[0]


def date_2_string(date):
    return "{}-{}-{}".format(date.day, date.month, date.year)


def remove_url(url):
    return re.sub(r"https?://\S+", "", url)


def request_from_url(url, headers):
    response = requests.get(url, headers=headers)
    return response.json()


def clean_text(text):
    emojis = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E6-\U0001F1FF"  # flags
                        u"\U0001F600-\U0001F64F"
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        u"\U0001f926-\U0001f937"
                        u"\U0001F1F2"
                        u"\U0001F1F4"
                        u"\U0001F620"
                        u"\u200d"
                        u"\u2640-\u2642"
                        "]+", flags=re.UNICODE)
    text = emojis.sub(r"", text)
    # eof = re.compile("\n")
    # text = eof.sub(r"", text)

    text = text.replace("\n", "")
    text = text.replace('"', "'")
    return text


################################################################################
################################################################################
############################JSON################################################
################################################################################
################################################################################
async def get_article(url, session):
    try:
        if url == "":
            return ""
        else:
            async with session.get(url, timeout=TIMEOUT) as response:
                if response.status == 200:
                    res = await response.text()
                    return res
            return ""
    except RuntimeError:
        logging.error("Error: RuntimeError")
    except ClientConnectorError:
        logging.error("Error: ClientConnectorError")
    except ClientOSError:
        logging.error("Error: ClientOSError")
    except UnicodeError:
        logging.error("Error: UnicodeError")
    except asyncio.exceptions.TimeoutError:
        logging.error("Timeout in {}".format(url))


async def get_articles(data):
    async with aiohttp.ClientSession() as session:
        urls = []
        texts = []

        for element in data:
            entities = element.get("entities")
            url = ""

            if entities is not None and entities.get("urls"):
                if entities["urls"][0].get("unwound_url"):
                    url = entities["urls"][0]["unwound_url"]
                else:
                    url = entities["urls"][0]["expanded_url"]

            urls.append(asyncio.ensure_future(get_article(url, session)))

        start = time.time()

        htmls = await asyncio.gather(*urls)
        logging.info("Get urls: {}".format(time.time() - start))
        goose = Goose()

        start = time.time()
        for html in htmls:
            try:
                if html is None:
                    texts.append(["", ""])
                else:
                    article = goose.extract(raw_html=str(html))
                    texts.append([article.title, article.cleaned_text, article.opengraph.get("image"), article.domain, article.publish_date])
            except TypeError as error:
                logging.error("Error: {}".format(html))
                texts.append(["", ""])
            except ParserError as error:
                logging.error("Error: ParserError")
                texts.append(["", ""])

        logging.info(time.time() - start)
        return texts


################################################################################
################################################################################
############################JSON################################################
################################################################################
################################################################################
async def json_2_tweet(json_response):
    data = json_response["data"]
    tweets = Tweets()
    languages = ["en"]  # "es",

    articles = await get_articles(data)

    for element, article in zip(data, articles):
        if (element["lang"] in languages) and not element.get("withheld"):
            interactions = 0
            conversation_id = element["id"]
            text = remove_url(element["text"])

            for i in element["public_metrics"]:
                interactions += element["public_metrics"][i]

            text = clean_text(text)
            title = clean_text(article[0])
            body = clean_text(article[1])

            date = ""
            domain = ""

            try:
                domain = article[3]
                date = parse_date_2(article[4])
            except IndexError:
                domain = ""
                date = ""
            except ValueError:
                date = article[4]
            except TypeError:
                date = ""

            try:
                image = article[2]
            except IndexError:
                image = ""

            tweet = Tweet(title, body, text, interactions, conversation_id, image, domain, date)

            if len(text) > 100:
                if body == "" or (date == "" and (domain is None or domain == "")) or (body != "" and len(body) < 100):
                    tweet.body = ""
                    tweet.title = ""
                    tweet.domain = "Unknown"
                    classification = pred(text)
                else:
                    classification = pred(body)

                tweet.truthfulness = str(round(classification[0][0], 3))

                tweets.push(tweet)

    return tweets


def json_2_trends(json_response):
    data = json_response[0]["trends"]
    trends = []

    for trend in data:
        name = trend["name"]
        tweet_volume = trend["tweet_volume"]
        trend_dict = {
            "name": name,
            "tweet_volume": tweet_volume
        }

        trends.append(trend_dict)

    return jsonify(trends)


def json_2_conversation(json_response):
    data = json_response["data"]
    texts = []

    for element in data:
        texts.append(clean_text(element["text"]))

    return jsonify(texts)


def json_2_counts(json_response, r):
    data = json_response["data"]
    counts = []
    i = 0
    n = 0
    r = int(r)
    hour_str = ""

    for count in data:
        if i == 0:
            hour = count["start"]
            hour = parse_date(hour)
            hour_str = date_2_string(hour)
            n = 0

        number_tweets = count["tweet_count"]
        n += number_tweets
        i += 1

        if i == r:
            count_dict = {
                "hour": hour_str,
                "tweets": n
            }
            counts.append(count_dict)
            i = 0

    return jsonify(counts)


def get_user_id(json_response):
    id = json_response["data"]["id"]
    return int(id)


def json_2_user(json_response):
    data = json_response["data"]
    tweets = []

    for element in data:
        tweets.append(element["text"])

    return jsonify(tweets)


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
def tokenize(tokenizer, element):
    return pad_sequences(tokenizer.texts_to_sequences(numpy.array([element])), maxlen=500)


def pred(element):
    return model_2.predict([tokenize(tokenizer, element)])
    #return model_2.predict_classes(tokenize(tokenizer, element))[0][0]
