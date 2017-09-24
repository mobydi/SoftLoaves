# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import re
import nltk
import codecs
import os

from tokenize_v2_ import Tokenizer
import pandas as pd


class DullModel:
    def __init__(self, path):

        benefits_voc = []

        self.tok = Tokenizer(path)

        liness_benefit = codecs.open(path + "/benefit_voc.txt", 'r', 'utf-8')

        for line in liness_benefit:
            line = line.strip()
            benefits_voc.append(line)

        liness_drawbacks = codecs.open(path + "/drawbacks_voc.txt", 'r',
                                       'utf-8')
        self.benefits_voc = np.array(benefits_voc)

        drawbacks_voc = []

        for line in liness_drawbacks:
            line = line.strip()
            drawbacks_voc.append(line)

        self.drawbacks_voc = np.array(drawbacks_voc)

    def clean_str_ru2(self, string, max_len=200):

        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        # string = re.sub(r"...", " ... ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        r = re.compile(u"[а-яА-Я]+")

        russian = r.findall(string)
        russian = [w.lower() for w in russian]

        if (len(russian) > max_len):
            russian = russian[:max_len + 1]

        russian = " ".join(russian)

        return self.tok.tokenize(russian)

    def raw_to_tokens(self, raw_text, max_len=400):

        text_coments = np.array(
            [self.clean_str_ru2(sent, max_len=max_len) for sent in raw_text])

        return text_coments

    def predict(self, data, work_ids):
        top_n = 7

        using_voc_ben = 10000
        using_voc_db = 10000
        slide_win_before = 3
        slide_win_after = 10

        text = np.array(data['TEXT'])

        likes_count = np.array(data['LIKES_COUNT'])
        dislikes_count = np.array(data['DISLIKES_COUNT'])
        recommends = np.array(data['RECOMMENDED'])

        text = text[work_ids]
        likes_count = likes_count[work_ids]
        dislikes_count = dislikes_count[work_ids]
        recommends = recommends[work_ids]

        text_tok = self.raw_to_tokens(text)

        drawbacks_voc = self.drawbacks_voc[:using_voc_db]
        benefits_voc = self.benefits_voc[:using_voc_ben]

        result = []
        rate = []

        for k, p in enumerate(text_tok):
            j = 0
            for i in range(len(p)):
                if (j >= len(p)):
                    break

                if (not (p[j] in drawbacks_voc) | (p[j] in benefits_voc)):
                    j += 1
                    continue

                v = p[max(0, j - slide_win_before):  min(j + slide_win_after,
                                                         len(p))]
                result.append("  ".join(v))
                j += slide_win_after
                pos_r = likes_count[k]
                if (pd.isnull(pos_r)):
                    pos_r = 0
                rec = recommends[k]
                if (pd.isnull(rec)):
                    rec = 0
                neg_r = dislikes_count[k]
                if (pd.isnull(neg_r)):
                    neg_r = 0
                neg_r *= -1.
                rate.append(pos_r + rec + neg_r)

        result = np.array(result)
        rate = np.array(rate)
        ids = np.argsort(rate)
        ids = ids[::-1]
        result = result[ids]
        result = result[:top_n]

        return result
