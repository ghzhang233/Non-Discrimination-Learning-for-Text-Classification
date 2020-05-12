import os

import numpy as np
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_predict

from data_utils import read_data, clean_text, get_sensitive_words

dir_processed = "./processed_data/"
if not os.path.isdir(dir_processed):
    os.mkdir(dir_processed)


def get_tfidf(X, sensitive_words):
    tf = np.zeros([len(X), len(sensitive_words)])
    idf = np.zeros(len(sensitive_words))
    tfidf = np.zeros([len(X), len(sensitive_words)])
    idxs_sens = []
    for i in range(len(X)):
        for j in range(len(sensitive_words)):
            tf[i, j] = sum([(1 if word == sensitive_words[j] else 0) for word in X[i].split()])
            if tf[i, j] > 0:
                idf[j] += 1
        if tf[i].sum() > 0:
            idxs_sens.append(i)
    for i in range(len(sensitive_words)):
        idf[i] = np.log(len(idxs_sens) / (idf[i] + 1))
    for i in range(len(X)):
        for j in range(len(sensitive_words)):
            tfidf[i, j] = tf[i, j] * idf[j]
    print(len(idxs_sens))
    return tfidf, idxs_sens


def make_weights(name_dataset):
    sensitive_words, sensitive_general_words, sensitive_extra_words = get_sensitive_words()
    sensitive_words = list(set(sensitive_words))
    data, split = read_data(name_dataset)

    X = data["text"].apply(clean_text).values
    label_encoder = LabelEncoder().fit(data["label"].values)
    y = label_encoder.transform(data["label"].values)

    X = X[np.concatenate([split[0], split[1]])]
    y = y[np.concatenate([split[0], split[1]])]

    Z, idxs_sens = get_tfidf(X, sensitive_words)

    # obtaining the weights
    clf = RandomForestClassifier(n_estimators=1000, max_depth=27, random_state=233, n_jobs=14, criterion='entropy')
    y_pred = cross_val_predict(clf, Z[idxs_sens], y[idxs_sens], cv=250, n_jobs=1, method='predict_proba')
    print('Refit log loss: %.5f' % (log_loss(y[idxs_sens], y_pred[:, 1])))

    p1 = sum(y[idxs_sens]) / len(y[idxs_sens])
    p0 = 1 - p1
    print(roc_auc_score(to_categorical(y[idxs_sens]), y_pred))
    print(accuracy_score(y[idxs_sens], np.argmax(y_pred, 1)), max(p0, p1))

    p1 = sum(y) / len(y)
    p0 = 1 - p1
    propensity = np.array([(p1 if y[i] == 1 else p0) for i in range(len(y))])
    propensity[idxs_sens] = np.array([y_pred[i, y[idxs_sens[i]]] for i in range(len(idxs_sens))])
    np.save(dir_processed + "propensity_%s.npy" % name_dataset, propensity)
    # propensity = np.load(dir_processed + "propensity_%s.npy" % name_dataset)

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
    make_weights("st")
