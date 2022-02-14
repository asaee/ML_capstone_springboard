import pytest

from src.data.make_dataset import ProcessText


@pytest.fixture
def get_processtext():
    textProcessor = ProcessText()

    return textProcessor


def test_strip_html_tags(get_processtext):
    assert get_processtext.strip_html_tags(
        '<html>Not a Tag</html>') == 'Not a Tag'


def test_remove_accented_chars(get_processtext):
    assert get_processtext.remove_accented_chars("éèâîôñüïç") == "eeaionuic"


def test_expand_contractions(get_processtext):
    assert get_processtext.expand_contractions(
        "It'd shouldn't aren't") == "It would should not are not"


def test_spell_correction(get_processtext):
    assert get_processtext.spell_correction(
        'All erors shold have beene fixde in thiss texxt') == 'All errors should have been fixed in this text'


def test_lemmatize_text(get_processtext):
    assert get_processtext.lemmatize_text(
        "These tests evaluated functions accuracies and the words are lemmatized") == \
        "these test evaluate function accuracy and the word be lemmatize"


def test_remove_stopwords(get_processtext):
    assert get_processtext.remove_stopwords(
        "There is no stopword in this text, not is not considered a stopword in this analysis") == \
        "no stopword text , not not considered stopword analysis"
