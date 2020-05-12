import os

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, roc_auc_score

import pandas as pd
import numpy as np
from data_utils import read_data, preprocess_data, read_eec, read_weights, get_augmentation_data, process_data
from config import Config
from models import get_model
from general_utils import false_positive_rate, false_negative_rate

dir_trained = "./trained_models/"
if not os.path.isdir(dir_trained):
    os.mkdir(dir_trained)

config = Config()
config.set_params_parser()

data, idxs = read_data(config.name_dataset, use_loaded=True)
idxs_train, idxs_dev, idxs_test = idxs
X, y, emb, tokenizer, label_encoder = preprocess_data(data=data, name_dataset=config.name_dataset, use_loaded=True)
X_aug_train, y_aug_train = get_augmentation_data(data.iloc[idxs_train], tokenizer, label_encoder)
X_aug_dev, y_aug_dev = get_augmentation_data(data.iloc[idxs_dev], tokenizer, label_encoder)
X_eec, y_eec, idxs_gender = read_eec(tokenizer, label_encoder)
debias_weights = read_weights(config.name_dataset)

acc_dev_list, auc_dev_list, acc_test_list, auc_test_list = [], [], [], []
acc_eec_list, auc_eec_list, efpr_eec_list, efnr_eec_list, eauc_eec_list = [], [], [], [], []
for round in range(config.round):
    model = get_model(emb, num_lstm=1)
    file_best_model = dir_trained + "best_model_%s_%d.h5" % (config.name_model, round)
    checkpoint = ModelCheckpoint(file_best_model, monitor='val_loss',
                                 save_best_only=True, mode='auto', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=1)
    if config.use_weights:
        history = model.fit(X[idxs_train], to_categorical(y[idxs_train]), sample_weight=debias_weights[idxs_train],
                            validation_data=[X[idxs_dev], to_categorical(y[idxs_dev]), debias_weights[idxs_dev]],
                            epochs=100, batch_size=256, callbacks=[checkpoint, early_stopping], verbose=1)
    else:
        if config.use_augmentation:
            history = model.fit(np.concatenate([X[idxs_train], X_aug_train]),
                                to_categorical(np.concatenate([y[idxs_train], y_aug_train])),
                                validation_data=[np.concatenate([X[idxs_dev], X_aug_dev]),
                                                 to_categorical(np.concatenate([y[idxs_dev], y_aug_dev]))],
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
    idxs_male, idxs_female = idxs_gender
    y_prob_test = model.predict(X_eec, batch_size=256)
    y_pred_test = np.argmax(y_prob_test, axis=1)
    acc_eec = accuracy_score(y_eec, y_pred_test)
    acc_eec_list.append(acc_eec)
    auc_eec = roc_auc_score(to_categorical(y_eec), y_prob_test)
    auc_eec_list.append(auc_eec)

    # efpr
    fpr_eec = false_positive_rate(y_eec, y_pred_test)
    fpr_male_eec = false_positive_rate(y_eec[idxs_male], y_pred_test[idxs_male])
    fpr_female_eec = false_positive_rate(y_eec[idxs_female], y_pred_test[idxs_female])
    # print(fpr_eec, fpr_male_eec, fpr_female_eec)
    efpr = abs(fpr_eec - fpr_male_eec) + abs(fpr_eec - fpr_female_eec)
    efpr_eec_list.append(efpr)

    # efnr
    fnr_eec = false_negative_rate(y_eec, y_pred_test)
    idxs_male, idxs_female = idxs_gender
    fnr_male_eec = false_negative_rate(y_eec[idxs_male], y_pred_test[idxs_male])
    fnr_female_eec = false_negative_rate(y_eec[idxs_female], y_pred_test[idxs_female])
    # print(fnr_eec, fnr_male_eec, fnr_female_eec)
    efnr = abs(fnr_eec - fnr_male_eec) + abs(fnr_eec - fnr_female_eec)
    efnr_eec_list.append(efnr)

    # eauc
    auc_eec = roc_auc_score(to_categorical(y_eec), y_prob_test)
    auc_male_eec = roc_auc_score(to_categorical(y_eec[idxs_male]), y_prob_test[idxs_male])
    auc_female_eec = roc_auc_score(to_categorical(y_eec[idxs_female]), y_prob_test[idxs_female])
    # print(auc_eec, auc_male_eec, auc_female_eec)
    eauc = abs(auc_eec - auc_male_eec) + abs(auc_eec - auc_female_eec)
    eauc_eec_list.append(eauc)

results = pd.DataFrame({"acc_dev": acc_dev_list,
                        "auc_dev": auc_dev_list,
                        "acc_test": acc_test_list,
                        "auc_test": auc_test_list,
                        "acc_eec": acc_eec_list,
                        "auc_eec": auc_eec_list,
                        "efpr_eec": efpr_eec_list,
                        "efnr_eec": efnr_eec_list,
                        "eauc_eec": eauc_eec_list})
print(results.describe())
results.to_csv("results_summary_%s.csv" % config.name_model)
