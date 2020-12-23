from collections import Counter

from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import LancasterStemmer, WordNetLemmatizer

STEMMER = None
LEMMATIZER = None
WORDS = None


def strip_html(text):
    '''get plain text from html
    '''
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def normalize_word(word, words=None, stemmer=None, lemmatizer=None):
    '''normalize_word to its stem form,
    if not a dictionary word, return ''
    '''
    # initialize once
    global WORDS, STEMMER, LEMMATIZER
    if words is None:
        if WORDS is None:
            WORDS = wordnet.words()
        words = WORDS
    if stemmer is None:
        if STEMMER is None:
            STEMMER = LancasterStemmer()
        stemmer = STEMMER
    if lemmatizer is None:
        if LEMMATIZER is None:
            LEMMATIZER = WordNetLemmatizer()
        lemmatizer = LEMMATIZER

    if word in words:
        return word
    temp = lemmatizer.lemmatize(word)
    if temp in words:
        return temp
    temp = lemmatizer.lemmatize(word, pos='v')
    if temp in wordnet.words():
        return temp
    temp = stemmer.stem(word)
    if temp in wordnet.words():
        return temp
    return ''


def text_to_normalized_words(text):
    '''``text``: a string
    return a Counter (multiset) of words, which are

    - HTML stripped
    - lower-case
    - normalized to stem words
    - remove non-dictionary words
    '''
    text = strip_html(text)
    text = text.lower()
    result = word_tokenize(text)
    result = [normalize_word(word) for word in result if word not in stopwords.words()]
    result = [word for word in result if word and word not in stopwords.words()]
    return Counter(result)
