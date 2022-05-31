import pandas
import re
import numpy
from bs4 import BeautifulSoup
from keras.saving.model_config import model_from_json
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow_hub
import tensorflow
import matplotlib.pyplot as plot
import tensorflow_text

GLOVE = "../input/glove-twitter/glove.twitter.27B.100d.txt"
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

    real_data["text"] = real_data["title"] + " " + real_data["text"]
    fake_data["text"] = fake_data["title"] + " " + fake_data["text"]

    real_data = pandas.DataFrame(real_data["text"])
    fake_data = pandas.DataFrame(fake_data["text"])

    real_data["fake"] = 0
    fake_data["fake"] = 1

    data = pandas.concat([real_data, fake_data])
    data["text"] = data["text"].apply(clean_data)
    data = shuffle(data)

    data.replace("", numpy.nan, inplace=True)
    data.replace(" ", numpy.nan, inplace=True)
    data.replace("  ", numpy.nan, inplace=True)
    data.dropna(inplace=True)

    return data


def plot_results(y_pred, y_test, hist):
    disp = ConfusionMatrixDisplay.from_predictions(y_pred.astype(int), y_test.astype(int), cmap=plot.cm.Blues,
                                                   normalize="true",
                                                   display_labels=["0", "1"])
    plot.savefig("confussion_matrix.png")
    plot.show()


def create_model():
    x_train, x_test, y_train, y_test = train_test_split(data.text, data.fake)

    bert_preprocess = tensorflow_hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = tensorflow_hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

    text_input = tensorflow.keras.layers.Input(shape=(), dtype=tensorflow.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    new_layer = tensorflow.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    new_layer = tensorflow.keras.layers.Dense(1, activation='sigmoid', name="output")(new_layer)

    model = tensorflow.keras.Model(inputs=[text_input], outputs=[new_layer])

    METRICS = [
        tensorflow.keras.metrics.BinaryAccuracy(name='accuracy'),
        tensorflow.keras.metrics.Precision(name='precision'),
        tensorflow.keras.metrics.Recall(name='recall'),
    ]

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=METRICS)

    hist = model.fit(x_train, y_train, epochs=2)
    accuracy = model.evaluate(numpy.array(x_test), numpy.array(y_test, numpy.float32))
    pred = model.predict(x_test)

    y_pred = (pred > 0.5).astype(int).ravel()

    y_test = numpy.array(y_test, numpy.float32).ravel()
    print(classification_report(y_test, y_pred))

    plot_results(y_pred, y_test, hist)

    print("Accuracy: " + str(round(accuracy[1], 3)))
    print("% error test: " + str(round(1.0 - accuracy[1], 3)))

    save_model(model)


def save_model(model):
    model_json = model.to_json()
    with open("model_3/model_3.json", "w") as f:
        f.write(model_json)

    model.save_weights("model_3/model_3.h5")


def load_model():
    model_json_r = open("model_3/model_3.json", "r")
    model_json = model_json_r.read()
    model_json_r.close()

    model = model_from_json(model_json, custom_objects={'KerasLayer': tensorflow_hub.KerasLayer})
    model.load_weights("model_3/model_3.h5")

    return model


data = load_data()
create_model()
# model = load_model()
