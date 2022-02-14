# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from bs4 import BeautifulSoup as bs
import unicodedata
import re
from contractions import contractions_dict
from textblob import TextBlob
import nltk
import spacy
from nltk.tokenize.toktok import ToktokTokenizer
import pandas as pd


class ProcessText:
    def __init__(self) -> None:
        # Contraction mapping
        self.contractions = contractions_dict

    def strip_html_tags(self, text: str) -> str:
        soup = bs(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    def remove_accented_chars(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def remove_special_characters(self, text: str, remove_digits: bool = False) -> str:
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text

    def expand_contractions(self, text):
        contraction_mapping = self.contractions
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)\
                if contraction_mapping.get(match)\
                else contraction_mapping.get(match.lower())
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction

        try:
            expanded_text = contractions_pattern.sub(expand_match, text)
            expanded_text = re.sub("'", "", expanded_text)
        except:
            return text

        return expanded_text

    def spell_correction(self, text):
        tb = TextBlob(text)
        fixed_text = tb.correct()
        return str(fixed_text)

    def remove_incomprehensible_words(self, text):
        en_words = set(nltk.corpus.words.words())
        text = " ".join(w for w in nltk.wordpunct_tokenize(text)
                        if w.lower() in en_words or not w.isalpha())
        return text

    def lemmatize_text(self, text):
        nlp = spacy.load('en_core_web_sm')
        nlp.max_length = 1500000
        text = nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ !=
                        '-PRON-' else word.text for word in text])
        return text

    def remove_stopwords(self, text, is_lower_case=False):
        stopword_list = nltk.corpus.stopwords.words('english')
        stopword_list.remove('no')
        stopword_list.remove('not')
        tokenizer = ToktokTokenizer()

        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [
                token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [
                token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def convert_text(self, text, case='lower'):
        if case == 'lower':
            text = text.lower()
        elif case == 'upper':
            text = text.upper()
        elif case == 'title':
            text = text.title()
        else:
            text
        return text

    def normalize_corpus(self, corpus, html_stripping=True, contraction_expansion=True,
                         accented_char_removal=True, text_lower_case=True,
                         text_lemmatization=True, special_char_removal=True,
                         stopword_removal=True, remove_digits=True,
                         correct_spelling=False, remove_incomprehensible_word=True):

        normalized_corpus = []
        # normalize each document in the corpus
        for doc in corpus:
            # strip HTML
            if html_stripping:
                doc = self.strip_html_tags(doc)
            # remove accented characters
            if accented_char_removal:
                doc = self.remove_accented_chars(doc)
            # expand contractions
            if contraction_expansion:
                doc = self.expand_contractions(doc, self.contractions)
            # lowercase the text
            if text_lower_case:
                doc = doc.lower()
            # remove extra newlines
            doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
            # lemmatize text
            if text_lemmatization:
                doc = self.lemmatize_text(doc)
            # remove special characters and\or digits
            if special_char_removal:
                # insert spaces between special characters to isolate them
                special_char_pattern = re.compile(r'([{._(-)!}])')
                doc = special_char_pattern.sub(" \\1 ", doc)
                doc = self.remove_special_characters(
                    doc, remove_digits=remove_digits)
            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)
            # remove stopwords
            if stopword_removal:
                doc = self.remove_stopwords(doc, is_lower_case=text_lower_case)
            # correct spelling
            if correct_spelling:
                doc = self.spell_correction(doc)
            # remove incomprehensible words
            if remove_incomprehensible_word:
                doc = self.remove_incomprehensible_words(doc)

            normalized_corpus.append(doc)

        return normalized_corpus


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
     Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed(saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    df = pd.read_csv(input_filepath)
    text_normalier = ProcessText()
    normal_corpus = text_normalier.normalize_corpus(df['content'].values)
    normal_corpus.to_csv(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
