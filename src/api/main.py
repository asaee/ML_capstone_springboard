import pickle5 as pickle
from keras.models import load_model
from fastapi import FastAPI
from src.models.predict_model import *


def load_model_tokenizer():
    path_tokenizer = "../lib/tokenizer.pickle"
    with open(path_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)

    path_model = "../lib/cnn_model"
    model = load_model(path_model, compile=False)

    return tokenizer, model


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
