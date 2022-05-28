
class Tweets:

    def __init__(self):
        self.tweets = []

    def push(self, tweet):
        self.tweets.append(tweet.to_json())

    def get(self, i):
        return self.tweets[i]

    def to_json(self):
        return self.__dict__
