import os
from collections import defaultdict

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, roc_auc_score

import pandas as pd
import numpy as np
from data_utils import read_data, preprocess_data, read_eec, read_weights, process_data, read_official_test, \
    get_supplementation
from config import Config
from models import get_model
from general_utils import false_positive_rate, false_negative_rate

dir_trained = "./trained_models/"
dir_results = "./results/"
if not os.path.isdir(dir_trained):
    os.mkdir(dir_trained)
if not os.path.isdir(dir_results):
    os.mkdir(dir_results)

config = Config()
config.set_params_parser()

data, idxs = read_data(use_loaded=True)
idxs_train, idxs_dev, idxs_test = idxs
X, y, emb, tokenizer, label_encoder = preprocess_data(data=data, use_loaded=True)
X_sup_train, y_sup_train = process_data(get_supplementation(data.iloc[idxs_train], 'train', use_loaded=True), tokenizer,
                                        label_encoder, max_len_seq=35)
X_sup_dev, y_sup_dev = process_data(get_supplementation(data.iloc[idxs_dev], 'dev', use_loaded=True), tokenizer,
                                    label_encoder, max_len_seq=35)
X_eec, y_eec, idxs_identity = read_eec(tokenizer, label_encoder)
debias_weights = read_weights()

data_official_test = read_official_test()
X_official_test, _ = process_data(data_official_test, tokenizer, label_encoder)

acc_dev_list, auc_dev_list, acc_test_list, auc_test_list = [], [], [], []
acc_eec_list, auc_eec_list, efpr_eec_list, efnr_eec_list, eauc_eec_list = [], [], [], [], []
fpr_eec_dict, fnr_eec_dict, auc_eec_dict = defaultdict(int), defaultdict(int), defaultdict(int)
for round in range(config.round):
    model = get_model(emb, use_cudnn=(not config.not_use_cudnn), num_lstm=1)
    file_best_model = dir_trained + "best_model_%s_%d.h5" % (config.name_model, round)
    checkpoint = ModelCheckpoint(file_best_model, monitor='val_loss',
                                 save_best_only=True, mode='auto', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=1)
    if config.use_weights:
        history = model.fit(X[idxs_train], to_categorical(y[idxs_train]), sample_weight=debias_weights[idxs_train],
                            validation_data=[X[idxs_dev], to_categorical(y[idxs_dev]), debias_weights[idxs_dev]],
                            epochs=100, batch_size=256, callbacks=[checkpoint, early_stopping], verbose=1)
    else:
        if config.use_supplementation:
            history = model.fit(np.concatenate([X[idxs_train], X_sup_train]),
                                to_categorical(np.concatenate([y[idxs_train], y_sup_train])),
                                validation_data=[np.concatenate([X[idxs_dev], X_sup_dev]),
                                                 to_categorical(np.concatenate([y[idxs_dev], y_sup_dev]))],
                                epochs=100, batch_size=256, callbacks=[checkpoint, early_stopping], verbose=1)
        else:
            history = model.fit(X[idxs_train], to_categorical(y[idxs_train]),
                                validation_data=[X[idxs_dev], to_categorical(y[idxs_dev])],
                                epochs=100, batch_size=256, callbacks=[checkpoint, early_stopping], verbose=1)
    model.load_weights(file_best_model)

    # acc/auc dev
    y_prob_dev = model.predict(X[idxs_dev], batch_size=256)
    y_pred_dev = np.argmax(y_prob_dev, axis=1)
    acc_dev = accuracy_score(y[idxs_dev], y_pred_dev)
    acc_dev_list.append(acc_dev)
    auc_dev = roc_auc_score(to_categorical(y[idxs_dev]), y_prob_dev)
    auc_dev_list.append(auc_dev)

    # acc/auc test
    y_prob_test = model.predict(X[idxs_test], batch_size=256)
    y_pred_test = np.argmax(y_prob_test, axis=1)
    acc_test = accuracy_score(y[idxs_test], y_pred_test)
    acc_test_list.append(acc_test)
    auc_test = roc_auc_score(to_categorical(y[idxs_test]), y_prob_test)
    auc_test_list.append(auc_test)

    # auc/acc eec
    y_prob_test = model.predict(X_eec, batch_size=256)
    y_pred_test = np.argmax(y_prob_test, axis=1)
    acc_eec = accuracy_score(y_eec, y_pred_test)
    acc_eec_list.append(acc_eec)
    auc_eec = roc_auc_score(to_categorical(y_eec), y_prob_test)
    auc_eec_list.append(auc_eec)

    # efpr
    fpr_eec = false_positive_rate(y_eec, y_pred_test)
    fpr_eec_dict["all"] += fpr_eec
    efpr = 0
    for i in idxs_identity.keys():
        fpr_identity_eec = false_positive_rate(y_eec[idxs_identity[i]], y_pred_test[idxs_identity[i]])
        fpr_eec_dict[i] += fpr_identity_eec
        efpr += abs(fpr_eec - fpr_identity_eec)
    efpr_eec_list.append(efpr)

    # efnr
    fnr_eec = false_negative_rate(y_eec, y_pred_test)
    fnr_eec_dict["all"] += fnr_eec
    efnr = 0
    for i in idxs_identity.keys():
        fnr_identity_eec = false_negative_rate(y_eec[idxs_identity[i]], y_pred_test[idxs_identity[i]])
        fnr_eec_dict[i] += fnr_identity_eec
        efnr += abs(fnr_eec - fnr_identity_eec)
    efnr_eec_list.append(efnr)

    # eauc
    auc_eec = roc_auc_score(to_categorical(y_eec), y_prob_test)
    auc_eec_dict["all"] += auc_eec
    eauc = 0
    for i in idxs_identity.keys():
        auc_identity_eec = roc_auc_score(y_eec[idxs_identity[i]], y_pred_test[idxs_identity[i]])
        auc_eec_dict[i] += auc_identity_eec
        eauc += abs(auc_eec - auc_identity_eec)
    eauc_eec_list.append(eauc)

    # official test
    y_prob_test = model.predict(X_official_test, batch_size=256)
    data_official_test["prediction"] = y_prob_test[:, 1]
    data_official_test[["id", "prediction"]].to_csv(
        dir_results + "official_test_results_%s_%d.csv" % (config.name_model, round), index=False)

ideneity_keys = list(fpr_eec_dict.keys())
pd.DataFrame({"identity": ideneity_keys,
              "fpr": [fpr_eec_dict[i] / config.round for i in ideneity_keys],
              "fnr": [fnr_eec_dict[i] / config.round for i in ideneity_keys],
              "auc": [auc_eec_dict[i] / config.round for i in ideneity_keys]
              }).to_csv(dir_results + "results_identity_%s.csv" % config.name_model)

results = pd.DataFrame({"acc_dev": acc_dev_list,
                        "auc_dev": auc_dev_list,
                        "acc_test": acc_test_list,
                        "auc_test": auc_test_list,
                        "acc_eec": acc_eec_list,
                        "auc_eec": auc_eec_list,
                        "efpr_eec": efpr_eec_list,
                        "efnr_eec": efnr_eec_list,
                        "eauc_eec": eauc_eec_list})
results.to_csv(dir_results + "results_summary_%s.csv" % config.name_model)
print(results.describe())
