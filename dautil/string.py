from collections import Counter

from bs4 import BeautifulSoup
from nltk.corpus import wordnet, stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk import word_tokenize

STEMMER = LancasterStemmer()
LEMMATIZER = WordNetLemmatizer()
WORDS = wordnet.words()


def strip_html(text):
    '''get plain text from html
    '''
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def normalize_word(word, words=WORDS, stemmer=STEMMER, lemmatizer=LEMMATIZER):
    '''normalize_word to its stem form,
    if not a dictionary word, return ''
    '''
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
