import pickle
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI


def load_model_tokenizer():
    path_tokenizer = "./lib/tokenizer.pickle"
    with open(path_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)

    path_model = "./lib/model/cnn_model_config.json"
    # load json and create model
    json_file = open(path_model, 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("./lib/model/cnn_model_weights.h5")

    return tokenizer, loaded_model


def process_input_text(corpus, tokenizer, maxlen: int = 700):
    corpus = tokenizer.texts_to_sequences(corpus)
    corpus = pad_sequences(corpus, padding='post', maxlen=maxlen)

    return corpus


def predict_label_prob(corpus, model_cnn):
    label_pred_prob = model_cnn.predict(corpus)

    return label_pred_prob


app = FastAPI()
TOKENIZER, MODEL_CNN = load_model_tokenizer()


@app.get("/")
def read_root():
    return "Please send your inquiry to the /article/?text= endpoint"


@app.get("/article/")
async def read_text(text: str):
    corpus = process_input_text([text], TOKENIZER)
    label_prob = predict_label_prob(corpus, MODEL_CNN).tolist()

    label_list = ['business', 'finance', 'general', 'science', 'tech']
    label = []
    for i in range(len(label_prob)):
        label.append(dict(zip(label_list, label_prob[i])))

    return label
