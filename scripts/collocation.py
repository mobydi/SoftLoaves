import numpy as np
import nltk
from nltk.collocations import *
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from gensim.summarization import summarize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.summarization.textcleaner import clean_text_by_sentences
from nltk.stem.porter import PorterStemmer
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec

stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = text.split(" ")
    stems = stem_tokens(tokens, stemmer)
    return stems


def clean_data(x):
    text = str(x['TEXT']).strip().replace('\n', '').replace('\\', '').replace(
        '\t', '')
    text = str(text).replace('\n', '')
    cl = clean_text_by_sentences(text)
    tt = ""
    for i in range(len(cl)):
        tt = tt + " " + str(
            str(cl[i]).split('Processed unit:')[1]).strip().replace("'", "")
    return tt


class Collocation_find():
    def __init__(self, stop_w: str, path='', ):
        df1 = pd.read_csv(path)
        self.st_tfidf = np.array(pd.read_csv(stop_w).stop_w)
        df1.TEXT = df1.apply(clean_data, axis=1)
        self.data = df1

    def predict(self, idc=''):
        text = ""
        for t in self.data[self.data.PRODUCT == idc].TEXT:
            text = text + " " + str(t)

        tokens = nltk.wordpunct_tokenize(text)
        for t in tokens:
            if t in self.st_tfidf or t == '' or t == 'nan':
                tokens.remove(t)

        finder = TrigramCollocationFinder.from_words(tokens, window_size=3)
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        scored = finder.score_ngrams(trigram_measures.poisson_stirling)

        res = []
        for i in range(min(20, len(scored))):
            res.append(' '.join(scored[0:min(20, len(scored))][i][0]))

        return res
