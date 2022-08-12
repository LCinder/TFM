import json

from tensorflow import keras
from keras_preprocessing.text import tokenizer_from_json
#from tensorflow_hub import KerasLayer
from keras.models import model_from_json
#import tensorflow_text


def load_model_bert():
    model_json_r = open("./model/model_4/model.json", "r")
    model_json = model_json_r.read()
    model_json_r.close()

    model = model_from_json(model_json, custom_objects={'KerasLayer': KerasLayer})
    model.load_weights("./model/model_4/model.h5")

    return model


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
