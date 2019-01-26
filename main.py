import json
import numpy as np


def get_match_for_pos_neg(truth, pred, truth_state=True, match=True):
    all_state_pos = np.where(truth == 1)[0] if truth_state else np.where(truth == 0)[0]
    count = 0
    for v in all_state_pos:
        if match and pred[v] == truth[v]:
            count += 1
        elif not match and pred[v] != truth[v]:
            count += 1
    return count


def get_pos_neg_results_from_threshold(y_pred, y, value_type, threshold):
    diff = 0.5 - threshold
    y_pred_new = np.round(np.clip(np.add(np.asarray(y_pred), diff), a_min=0, a_max=1))
    if value_type == 'true pos':
        value = get_match_for_pos_neg(truth=y, pred=y_pred_new, truth_state=True, match=True)
    elif value_type == 'true neg':
        value = get_match_for_pos_neg(truth=y, pred=y_pred_new, truth_state=False, match=True)
    elif value_type == 'false pos':
        value = get_match_for_pos_neg(truth=y, pred=y_pred_new, truth_state=False, match=False)
    elif value_type == 'false neg':
        value = get_match_for_pos_neg(truth=y, pred=y_pred_new, truth_state=True, match=False)
    return value


def get_precision(y_pred, y, threshold):
    true_pos = get_pos_neg_results_from_threshold(y_pred=y_pred, y=y, value_type='true pos', threshold=threshold)
    false_pos = get_pos_neg_results_from_threshold(y_pred=y_pred, y=y, value_type='false pos', threshold=threshold)
    return true_pos / (true_pos + false_pos)  # given all pred pos


def get_recall(y_pred, y, threshold):
    true_pos = get_pos_neg_results_from_threshold(y_pred=y_pred, y=y, value_type='true pos', threshold=threshold)
    false_neg = get_pos_neg_results_from_threshold(y_pred=y_pred, y=y, value_type='false neg', threshold=threshold)
    return true_pos / (true_pos + false_neg)  # given all ground truth pos


with open("./predictions_and_labels.json/predictions_and_labels.json") as f:
    data = json.load(f)

thresholds = np.arange(0.0, 1, 0.05)
results_threshold = {}

y_truth = data['Y']
data['Y_pred'] = np.asarray(data['Y_pred'])
# one hot encoding
y_truth_one_hot = np.zeros_like(data['Y_pred'])
for i, v in enumerate(y_truth):
    if v != []:
        for e in v:
            y_truth_one_hot[i][e - 1] = 1

label_results = []
for i in range(3):
    # label wise information
    tpos, tneg, fpos, fneg, precision, recall = [], [], [], [], [], []
    for t in thresholds:
        tpos.append(get_pos_neg_results_from_threshold(y_pred=data['Y_pred'][:, i], y=y_truth_one_hot[:, i],
                                                       value_type='true pos', threshold=t))
        tneg.append(get_pos_neg_results_from_threshold(y_pred=data['Y_pred'][:, i], y=y_truth_one_hot[:, i],
                                                       value_type='true neg', threshold=t))
        fpos.append(get_pos_neg_results_from_threshold(y_pred=data['Y_pred'][:, i], y=y_truth_one_hot[:, i],
                                                       value_type='false pos', threshold=t))
        fneg.append(get_pos_neg_results_from_threshold(y_pred=data['Y_pred'][:, i], y=y_truth_one_hot[:, i],
                                                       value_type='false neg', threshold=t))
        precision.append(get_precision(y_pred=data['Y_pred'][:, i], y=y_truth_one_hot[:, i], threshold=t))
        recall.append(get_recall(y_pred=data['Y_pred'][:, i], y=y_truth_one_hot[:, i], threshold=t))

    label_results.append({'true pos': tpos, 'true neg': tneg, 'false pos': fpos, 'false neg': fneg,
                          'precision': precision, 'recall': recall})

for l in label_results:
    asd = 5


def threshold_optimizer(y_pred, y, key, limit, t=[0, 1]):
    _s = lambda threshold, type: get_pos_neg_results_from_threshold(y_pred, y, value_type=type, threshold=threshold)

    demon = np.sum(y)
    t = [t[0], (t[0] + t[1]) / 2, t[1]]
    if key.__contains__('pos') or key == 'recall':  # decreasing relationship
        key_rate = [_s(x, 'true pos') / (_s(x, 'true pos') + _s(x, 'false neg')) for x in
                    t] if key == 'recall' else [_s(x, key) / demon for x in t]
        if key_rate[0] > limit and limit > key_rate[1]:
            t[2] = (t[2] + t[1]) / 2
        elif key_rate[1] > limit and limit > key_rate[2]:
            t[0] = (t[1] + t[0]) / 2
    elif key.__contains__('neg') or key == 'precision':  # increasing relationship

        key_rate = [_s(x, 'true pos') / (_s(x, 'true pos') + _s(x, 'false pos')) for x in
                    t] if key == 'precision' else [_s(x, key) / demon for x in t]
        if key_rate[0] < limit and limit < key_rate[1]:
            t[2] = (t[2] + t[1]) / 2
        elif key_rate[1] < limit and limit < key_rate[2]:
            t[0] = (t[1] + t[0]) / 2

    if t[2] - t[0] < 0.00001:
        return t[1]
    return threshold_optimizer(y_pred, y, key, limit, t=[t[0], t[2]])


false_pos_20_thresholds = []
recall_80_thresholds = []
for label in range(3):
    false_pos_20_thresholds.append(threshold_optimizer(y_pred=[x[label] for x in data['Y_pred']],
                                                       y=y_truth_one_hot[:, label], key='false pos', limit=0.2))
    recall_80_thresholds.append(threshold_optimizer(y_pred=[x[label] for x in data['Y_pred']],
                                                    y=y_truth_one_hot[:, label], key='recall', limit=0.8))

true_pos_at_false_pos_20 = [
    get_pos_neg_results_from_threshold(y_pred=data['Y_pred'][:, i], y=y_truth_one_hot[:, i], value_type='true pos',
                                       threshold=v) / np.sum(y_truth_one_hot[:, i]) for i, v in
    enumerate(false_pos_20_thresholds)]
precision_at_recall_80 = [get_precision(y_pred=data['Y_pred'][:, i], y=y_truth_one_hot[:, i], threshold=v) for i, v in
                          enumerate(recall_80_thresholds)]

import pandas as pd

qn2_results = pd.DataFrame(
    {'labels': [1, 2, 3], 'Minimum threshold at maximum 0.2 false positive rate': false_pos_20_thresholds,
     'Minimum true positive rate at maximum 0.2 false positive rate': true_pos_at_false_pos_20})
qn3_results = pd.DataFrame({'labels': [1, 2, 3], 'Maximum threshold at minimum 0.8 recall': recall_80_thresholds,
                            'Maximum precision at minimum 0.8 recall': precision_at_recall_80})
