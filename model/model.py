from keras.models import model_from_json


def load_model_bert():
    model_json_r = open("model_3/model_2.json", "r")
    model_json = model_json_r.read()
    model_json_r.close()

    model = model_from_json(model_json)
    model.load_weights("model_3/model_2.h5")

    return model
