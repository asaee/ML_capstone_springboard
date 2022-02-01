import logging
import numpy as np
import numpy.typing as npt
from keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO)


def process_input_text(corpus: npt.NDArray[str], tokenizer, maxlen: int = 700) -> npt.NDArray[int]:
    corpus = tokenizer.texts_to_sequences(corpus)
    corpus = pad_sequences(corpus, padding='post', maxlen=maxlen)

    return corpus


def predict_label_prob(corpus: npt.NDArray[int], model_cnn) -> npt.NDArray[float]:
    label_pred_prob = model_cnn.predict(corpus)
    return label_pred_prob


def predict_label(label_pred_prob: npt.NDArray[float]) -> npt.NDArray[str]:
    lablel_map = {0: 'business', 1: 'finance',
                  2: 'general', 3: 'science', 4: 'tech'}

    label_pred = np.vectorize(lablel_map.get)(
        np.argmax(label_pred_prob, axis=1))

    return label_pred
