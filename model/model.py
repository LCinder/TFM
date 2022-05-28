import pandas
import re
import numpy
import tensorflow_hub
from bs4 import BeautifulSoup
import keras.preprocessing as preprocessing
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential, model_from_json
from keras_preprocessing.text import tokenizer_from_json
from sklearn.model_selection import train_test_split
import json
import tensorflow_text

GLOVE = "./data/glove/glove.twitter.27B.100d.txt"
FEATURES = 10000


def clean_data(text):
    text = re.sub(r"http\S+", "", text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"pic.twitter.com/\S+", "", text)
    text = re.sub(r"[VIDEO]", "", text)
    text = re.sub(r"\S+(Reuters)", "", text)
    return text


def load_data():
    real_data = pandas.read_csv("data/True.csv")
    fake_data = pandas.read_csv("data/Fake.csv")

    # real_data["text"] = real_data["title"] + " " + real_data["text"]
    # fake_data["text"] = fake_data["title"] + " " + fake_data["text"]

    real_data = pandas.DataFrame(real_data["text"])
    fake_data = pandas.DataFrame(fake_data["text"])

    real_data["fake"] = 0
    fake_data["fake"] = 1

    data = pandas.concat([real_data, fake_data])
    data["text"] = data["text"].apply(clean_data)

    data.replace("", numpy.nan, inplace=True)
    data.replace(" ", numpy.nan, inplace=True)
    data.replace("  ", numpy.nan, inplace=True)
    data.dropna(inplace=True)

    return data


def text_2_token(data):
    tokenizer = preprocessing.text.Tokenizer(num_words=FEATURES)
    tokenizer.fit_on_texts(data.text)
    x = preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(data.text), maxlen=500)

    embedding = dict(get_coefs(*g.rstrip().rsplit(" ")) for g in open(GLOVE, encoding="utf8"))
    embeddings = numpy.stack(embedding.values())
    embedding_mean = embeddings.mean()
    embedding_std = embeddings.std()
    word_index = tokenizer.word_index
    words_numbers = min(FEATURES, len(word_index))
    embedding_size = embeddings.shape[1]
    embedding_matrix = numpy.random.normal(embedding_mean, embedding_std, (words_numbers, embedding_size))

    for word, i in word_index.items():
        embedding_vector = embedding.get(word)
        if embedding_vector is not None and i < FEATURES:
            embedding_matrix[i] = embedding_vector

    x_train, x_test, y_train, y_test = train_test_split(x, data.fake)

    model = Sequential([
        Embedding(FEATURES, output_dim=100, weights=[embedding_matrix], input_length=500, trainable=False),
        LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        LSTM(64, dropout=0.3, recurrent_dropout=0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train, validation_data=[x_test, y_test], epochs=2, batch_size=256)

    print("Accuracy Train: {}".format(model.evaluate(x_train, y_train)[1]))
    print("Accuracy Test: {}".format(model.evaluate(x_test, y_test)[1]))


def get_coefs(word, *args):
    return word, numpy.asarray(args, dtype="float32")


def load_model():
    model_json_r = open("model/model_2/model.json", "r")
    model_json = model_json_r.read()
    model_json_r.close()

    model = model_from_json(model_json)
    model.load_weights("model/model_2/model.h5")

    with open("model/model_2/data.txt", "r") as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
        f.close()

    return model, tokenizer


def load_model_bert():
    model_json_r = open("model/model_2.json", "r")
    model_json = model_json_r.read()
    model_json_r.close()

    model = model_from_json(model_json, custom_objects={'KerasLayer': tensorflow_hub.KerasLayer})
    model.load_weights("model/model_2.h5")

    return model
