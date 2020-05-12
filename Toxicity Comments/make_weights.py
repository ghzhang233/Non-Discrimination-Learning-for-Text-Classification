import re
import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_predict

dir_processed = "./processed_data/"
if not os.path.isdir(dir_processed):
    os.mkdir(dir_processed)


def read_identity_terms(identity_terms_path):
    with open(identity_terms_path) as f:
        return [term.strip() for term in f.readlines()]


madlibs_terms = read_identity_terms('bias_madlibs_data/adjectives_people.txt')
print(madlibs_terms)

SPLITS = ['train', 'dev']
wiki = {}
for split in SPLITS:
    wiki[split] = './data/wiki_%s.csv' % split
train_data = pd.read_csv(wiki['train'])
valid_data = pd.read_csv(wiki['dev'])
X = np.concatenate([train_data['comment'], valid_data['comment']], axis=0)
y = np.concatenate([train_data['is_toxic'], valid_data['is_toxic']], axis=0)
y = np.array(y, dtype="int32")

X = [i.lower() for i in X]
filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
maketrans = str.maketrans
translate_map = maketrans(filters, " " * len(filters))
X = [i.translate(translate_map) for i in X]
X = np.array([(" " + " ".join(i.strip().split()) + " ") for i in X])
sen_length = np.array([len(i.split()) for i in X])

use_loaded = True
if use_loaded:
    Z = np.load(dir_processed + "Z.npy")
else:
    names = [re.compile(" %s " % term) for term in madlibs_terms]
    Z = np.ones([len(X), len(names)]) * 0.1
    idxs = []
    for i in range(len(X)):
        for j in range(len(names)):
            Z[i, j] = len(names[j].findall(X[i]))
    np.save(dir_processed + "Z.npy", Z)

Z = np.concatenate([Z, sen_length[:, np.newaxis]], axis=1)
idxs_sens = [i for i in range(len(Z)) if Z[i, :-1].sum() != 0]

clf = RandomForestClassifier(n_estimators=100, max_depth=18, random_state=233, n_jobs=3, criterion='entropy')

y_pred = cross_val_predict(clf, Z, y, cv=5, n_jobs=1, method='predict_proba')
print('Refit log loss: %.5f' % (log_loss(y, y_pred[:, 1])))

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

np.save(dir_processed + "weights.npy", weights)
np.save(dir_processed + "weights_train.npy", weights[:len(train_data)])
np.save(dir_processed + "weights_dev.npy", weights[len(train_data):len(train_data) + len(valid_data)])
