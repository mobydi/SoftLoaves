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

    def filter_phrases_old(self, extracted_keywords,
                           test_product_category_name):
        assighned_phrases = dict()
        for sent in extracted_keywords:
            for word in self.tokenize(sent):
                idx, current_score = self.get_category_keyword_score(
                    test_product_category_name, word)
                if idx == -1:
                    continue
                if idx in assighned_phrases:
                    if current_score > assighned_phrases[idx][1]:
                        assighned_phrases[idx] = (sent, current_score)
                else:
                    assighned_phrases[idx] = (sent, current_score)
        return assighned_phrases

    def filter_phrases(self, extracted_keywords, test_product_category_name):
        assighned_phrases = dict()
        for sent in extracted_keywords:
            if 'купил' in sent:
                continue
            keywords = [
                self.get_category_keyword_score(test_product_category_name,
                                                word) for word in
                self.tokenize(sent)]
            idx, current_score = \
            sorted(keywords, key=lambda x: x[1], reverse=True)[0]
            if idx == -1:
                continue
            if idx in assighned_phrases:
                if current_score > assighned_phrases[idx][1]:
                    assighned_phrases[idx] = (sent, current_score)
            else:
                assighned_phrases[idx] = (sent, current_score)
        return assighned_phrases

    def predict(self, df, top_n=5):
        test_product_all_text = '.'.join(list(df['TEXT']))
        test_product_category_name = df['CATEGORY_NAME'].iloc[0]

        extracted_keywords = list()
        for sent in self.get_sentences(test_product_all_text):
            tokenized_sent = set(self.tokenize(sent))
            matches = self.minhash.query(self.get_hash(tokenized_sent))
            if len(matches) > 0 and len(tokenized_sent) > 1:
                extracted_keywords.append(sent.strip())
        if top_n != None and len(extracted_keywords) > top_n:
            filtered_phrases = self.filter_phrases(extracted_keywords,
                                                   test_product_category_name)
            filtered_phrases = [filtered_phrases[key][0] for key in
                                sorted(filtered_phrases.keys())]
            return filtered_phrases[:top_n]
        else:
            return extracted_keywords


def tag_init(path):
    category_keywords = pickle.load(open(path + 'category_keywords.pkl', 'rb'))
    lsh = pickle.load(open(path + 'minhash_lsh.pkl', 'rb'))
    stop_words = pickle.load(open(path + 'stop_words.pkl', 'rb'))
    return PhrasesExtractor(minhash=lsh,
                            category_keywords=category_keywords,
                            stop_words=stop_words)
