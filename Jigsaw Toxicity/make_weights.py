import os
import re

import numpy as np
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, make_scorer
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_predict

from data_utils import read_data, clean_text

dir_processed = "./processed_data/"
if not os.path.isdir(dir_processed):
    os.mkdir(dir_processed)


def read_identity_terms(identity_terms_path):
    with open(identity_terms_path) as f:
        return [term.strip() for term in f.readlines()]


def make_weights():
    data, split = read_data(use_loaded=True)
    with open("data/identity_columns.txt", "r", encoding="utf-8") as fin:
        identity_columns = [line.strip() for line in fin.readlines()]
    cleaned_text = data["text"].apply(clean_text).values
    Z = np.concatenate([data[i].values[:, np.newaxis] for i in identity_columns], axis=1)

    X = np.array([" " + i + " " for i in cleaned_text])
    madlibs_terms = read_identity_terms('data/adjectives_people.txt')
    names = [re.compile(" %s " % term) for term in madlibs_terms]
    Z2 = np.ones([len(X), len(names)]) * 0.1
    for i in range(len(X)):
        for j in range(len(names)):
            Z2[i, j] = len(names[j].findall(X[i]))
    Z = np.concatenate([Z, Z2], axis=1)
    Z = np.concatenate([Z, np.array([len(i.split()) for i in cleaned_text])[:, np.newaxis]], axis=1)

    np.save(dir_processed + "Z.npy", Z)
    # Z = np.load(dir_processed + "Z.npy")

    label_encoder = LabelEncoder().fit(data["label"].values)
    y = label_encoder.transform(data["label"].values)

    Z = Z[np.concatenate([split[0], split[1]])]
    y = y[np.concatenate([split[0], split[1]])]
    idxs_sens = [i for i in range(len(Z)) if Z[i, :-1].sum() != 0]

    # obtaining the weights
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=233, n_jobs=14, criterion='entropy')
    y_pred = cross_val_predict(clf, Z, y, cv=25, n_jobs=None, method='predict_proba')
    np.save(dir_processed + 'indexed_y_pred.npy', y_pred)
    print('Refit log loss: %.5f' % (log_loss(y, y_pred)))

    acc_maj = max(1 - sum(y) / len(y), sum(y) / len(y))
    print(roc_auc_score(to_categorical(y), y_pred))
    print(accuracy_score(y, np.argmax(y_pred, 1)), acc_maj)

    acc_maj = max(1 - sum(y[idxs_sens]) / len(y[idxs_sens]), sum(y[idxs_sens]) / len(y[idxs_sens]))
    print(roc_auc_score(to_categorical(y[idxs_sens]), y_pred[idxs_sens]))
    print(accuracy_score(y[idxs_sens], np.argmax(y_pred[idxs_sens], 1)), acc_maj)

    propensity = np.array([y_pred[i, y[i]] for i in range(len(y))])
    np.save(dir_processed + "propensity.npy", propensity)
    # propensity = np.load(dir_processed + "propensity.npy")

    weights = 1 / propensity
    a = np.mean(np.array([weights[i] for i in range(len(weights)) if y[i] == 0]))
    b = np.mean(np.array([weights[i] for i in range(len(weights)) if y[i] == 1]))
    print((1 / a) / (1 / a + 1 / b), (1 / b) / (1 / a + 1 / b))
    weights = np.array([(weights[i] / a if y[i] == 0 else weights[i] / b) for i in range(len(weights))])
    weights /= weights.mean()

    ret = np.zeros(len(data))
    ret[np.concatenate([split[0], split[1]])] = weights
    np.save(dir_processed + "weights.npy", ret)


if __name__ == '__main__':
    make_weights()
