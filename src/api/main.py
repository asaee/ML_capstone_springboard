from fastapi import FastAPI
from src.models.predict_model import *


app = FastAPI()


@app.get("/")
def read_root():
    return "Please send your inquiry to the /article/?text= endpoint"


@app.get("/article/")
async def read_text(text: str):
    corpus = process_input_text(
        [text], path_tokenizer="../lib/tokenizer.pickle")
    label_prob = predict_label_prob(
        corpus, path_model="../lib/cnn_model").tolist()

    label_list = ['business', 'finance', 'general', 'science', 'tech']
    label = []
    for i in range(len(label_prob)):
        label.append(dict(zip(label_list, label_prob[i])))

    return label
