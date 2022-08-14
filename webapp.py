from aioflask import Flask, jsonify, request
import controller
import requests
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

headers = requests.structures.CaseInsensitiveDict()
headers["Accept"] = "application/json"
headers["Authorization"] = "Bearer {}".format(os.getenv("TOKEN"))


@app.route("/")
def status():
    return jsonify("status: Ok")


@app.route("/term/<query>/<n>")
async def get_tweets(query, n):
    URL_SEARCH = controller.make_query(query, n, links=request.args.get("links"), order=request.args.get("order"))
    res_search = controller.request_from_url(URL_SEARCH, headers)
    tweets = await controller.json_2_tweet(res_search)
    return tweets.to_json()


@app.route("/trends")
def get_trends():
    URL_TRENDS = controller.search_trends()
    res_trends = controller.request_from_url(URL_TRENDS, headers)
    trends = controller.json_2_trends(res_trends)
    return trends


@app.route("/conversation/<id>")
def get_conversation(id):
    URL_CONVERSATION = controller.search_conversation(id)
    res_conversation = controller.request_from_url(URL_CONVERSATION, headers)
    conversation = controller.json_2_conversation(res_conversation)
    return conversation


@app.route("/counts/<query>/<n>")
def get_counts(query, n):
    URL_COUNTS = controller.search_counts(query)
    res_counts = controller.request_from_url(URL_COUNTS, headers)
    counts = controller.json_2_counts(res_counts, n)
    return counts


@app.route("/user/<username>")
def get_user(username):
    URL_USER = controller.search_user(username)
    res_user = controller.request_from_url(URL_USER, headers)
    user_id = controller.get_user_id(res_user)
    URL_USER_ID = controller.search_user_id(user_id)
    res_user_id = controller.request_from_url(URL_USER_ID, headers)
    user = controller.json_2_user(res_user_id)
    return user

