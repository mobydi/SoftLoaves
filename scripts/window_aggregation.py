import numpy as np


def words_to_line(words: list):
    return ''.join([w + ' ' for w in words])


def predict(s: str, model) -> float:
    prediction = model.predict([s])
    return prediction


def aggregate(comments: list, model, num_comments=3, wnd_l=5, stride=2,
              min_sentence_len=5,
              positive_q=(0.85, 0.95), negative_q=(0.05, 0.15),
              positive_threshold=4, negative_threshold=3):
    results = {}  # map comment_idx->[(index_in_comment, mark)]
    sentences = []
    for comment in comments:
        sentences.extend(
            [s for s in comment.split('.') if len(s) > min_sentence_len])

    for idx, comment in enumerate(sentences):
        words = [w.lower() for w in comment.split()]
        w_len = len(words)
        if w_len <= wnd_l:
            results[idx] = [(0, predict(comment, model))]
        else:
            results[idx] = []
            for i in range(0, w_len - wnd_l, stride):
                res = predict(words_to_line(words[i: i + wnd_l]), model)
                results[idx].append((i, res))

    prcnt_list = []
    for i in range(len(sentences)):
        for k in range(len(results[i])):
            prcnt_list.append(results[i][k][1])
    pos_qnt_l = np.percentile(prcnt_list, positive_q[0] * 100)
    pos_qnt_r = np.percentile(prcnt_list, positive_q[1] * 100)
    neg_qnt_l = np.percentile(prcnt_list, negative_q[0] * 100)
    neg_qnt_r = np.percentile(prcnt_list, negative_q[1] * 100)
    ret = {'+': [], '-': []}

    for idx in range(len(sentences)):
        for k in range(len(results[idx])):
            positive_skip = len(
                ret['+']) == num_comments or pos_qnt_l < positive_threshold
            negative_skip = len(
                ret['-']) == num_comments or neg_qnt_r > negative_threshold

            if positive_skip and negative_skip:
                break

            if pos_qnt_l <= results[idx][k][1] <= pos_qnt_r and not positive_skip:
                comment = sentences[idx]
                words = [w.lower() for w in comment.split()]
                start = results[idx][k][0]
                ret['+'].append(words_to_line(words[start: start + wnd_l]))
            if neg_qnt_l <= results[idx][k][1] <= neg_qnt_r and not negative_skip:
                comment = sentences[idx]
                words = [w.lower() for w in comment.split()]
                start = results[idx][k][0]
                ret['-'].append(words_to_line(words[start: start + wnd_l]))
    return ret
