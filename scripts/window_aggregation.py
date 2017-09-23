import numpy as np


def words_to_line(words: list):
    return ''.join([w + ' ' for w in words])


def predict(s: str, model) -> float:
    prediction = model.predict(s)
    return prediction


def aggregate(comments: list, model, q=(0.9, 0.95), wnd_l=5, stride=2):
    results = {}  # map comment_idx->[(index_in_comment, mark)]
    for idx, comment in enumerate(comments):
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
    for i in range(len(comments)):
        for k in range(len(results[i])):
            prcnt_list.append(results[i][k][1])
    qnt_l = np.percentile(prcnt_list, q[0] * 100)
    qnt_r = np.percentile(prcnt_list, q[1] * 100)
    ret = []
    for idx in range(len(comments)):
        for k in range(len(results[idx])):
            if qnt_l <= results[idx][k][1] <= qnt_r:
                comment = comments[idx]
                words = [w.lower() for w in comment.split()]
                start = results[idx][k][0]
                ret.append(words_to_line(words[start: start + wnd_l]))
    return ret
