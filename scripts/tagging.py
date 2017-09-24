import string, pickle
from datasketch import MinHash
import nltk


class PhrasesExtractor:
    def __init__(self, minhash, category_keywords, stop_words):
        self.category_keywords = category_keywords
        self.minhash = minhash
        self.stop_words = stop_words
        self.punctuation = string.punctuation + '»«-–—`\'()'

    def tokenize(self, file_text, use_lower=True, drop_stopwords=True,
                 use_stemmer=False, drop_numbers=False):
        tokens = nltk.word_tokenize(file_text)
        tokens = [x for x in tokens if (x not in self.punctuation)]

        if use_lower:
            tokens = [w.lower() for w in tokens]

        if drop_numbers:
            tokens = [x for x in tokens if (x not in string.digits)]

        if drop_stopwords:
            tokens = [x for x in tokens if (x not in self.stop_words)]

        if use_stemmer:
            pass

        return tokens

    def get_sentences(self, text):
        if '.' in text:
            for sent in text.split('.'):
                if ',' in sent:
                    for sent_2 in sent.split(','):
                        if ' и ':
                            for sent_3 in sent_2.split(' и '):
                                yield sent_3.strip()
                        else:
                            yield sent_2.strip()
                else:
                    yield sent.strip()
        else:
            yield text

    def get_hash(self, keywords):
        keywords = set(keywords)
        mhash = MinHash(num_perm=256)
        for item in keywords:
            mhash.update(item.encode('utf8'))
        return mhash

    def get_category_keyword_score(self, category, keyword):
        category = category.strip()
        assert (category in self.category_keywords)
        current_category_keywords = self.category_keywords[category]
        if keyword in current_category_keywords:
            index = list(current_category_keywords.keys()).index(keyword)
            score = current_category_keywords[keyword]
            return index, score
        else:
            return -1, 0

    def predict(self, df):
        test_product_all_text = '.'.join(list(df['TEXT']))
        test_product_category_name = df['CATEGORY_NAME'].iloc[0]

        extracted_keywords = list()
        for sent in self.get_sentences(test_product_all_text):
            tokenized_sent = set(self.tokenize(sent))
            matches = self.minhash.query(self.get_hash(tokenized_sent))
            if len(matches) > 0 and len(tokenized_sent) > 1:
                extracted_keywords.append(sent.strip())
        return extracted_keywords


def tag_init(path):
    category_keywords = pickle.load(open(path + 'category_keywords.pkl', 'rb'))
    lsh = pickle.load(open(path + 'minhash_lsh.pkl', 'rb'))
    stop_words = pickle.load(open(path + 'stop_words.pkl', 'rb'))
    return PhrasesExtractor(minhash=lsh,
                            category_keywords=category_keywords,
                            stop_words=stop_words)
