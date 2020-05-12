import os
import pickle
from collections import defaultdict

import numpy as np
import re
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer, tokenizer_from_json
from sklearn.preprocessing import LabelEncoder

dir_processed = "./processed_data/"
if not os.path.isdir(dir_processed):
    os.mkdir(dir_processed)


def get_sensitive_words():
    sensitive_general_words, sensitive_extra_words, sensitive_words = [], [], []
    with open("data/gender_general_swap_words.txt", "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            s = line.strip().split()
            if len(s) != 2:
                print(line)
                continue
            sensitive_general_words.append(clean_text(s[0]))
            sensitive_general_words.append(clean_text(s[1]))
    with open("data/gender_extra_swap_words.txt", "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            s = line.strip().split()
            if len(s) == 0:
                continue
            sensitive_extra_words.append(clean_text(s[0]))
            sensitive_extra_words.append(clean_text(s[1]))
    sensitive_words = sensitive_general_words + sensitive_extra_words
    return sensitive_words, sensitive_general_words, sensitive_extra_words


def read_data(name_dataset, use_loaded=True):
    file_data = dir_processed + "%s_origin.pkl" % name_dataset
    if use_loaded:
        data, idxs = pickle.load(open(file_data, "rb"))
        return data, idxs

    # read data
    data = pd.read_csv("data/%s.csv" % name_dataset)
    # data = data.rename(cols={""})
    if name_dataset == "abt":
        data["label"] = data["label"].apply(lambda x: "offensive" if x in ["abusive", "hateful"] else "non_offensive")
    else:
        data["label"] = data["label"].apply(lambda x: "offensive" if x in ["sexism"] else "non_offensive")

    num_samples = len(data)
    orders = np.random.permutation(num_samples)
    idxs_train = orders[0:int(num_samples * 0.8)]
    idxs_dev = orders[int(num_samples * 0.8): int(num_samples * 0.9)]
    idxs_test = orders[int(num_samples * 0.9):]
    idxs = (idxs_train, idxs_dev, idxs_test)

    pickle.dump((data, idxs), open(file_data, "wb"))
    return data, idxs


def preprocess_data(data,
                    name_dataset,
                    use_loaded=True,
                    file_emb="./data/glove.840B.300d.txt",
                    # file_emb="./data/GoogleNews-vectors-negative300-hard-debiased.txt",
                    max_num_words=50000,
                    max_len_seq=35,
                    emb_dim=300):
    # preprocess data
    file_processed_data = dir_processed + "%s_processed.pkl" % name_dataset
    file_tokenizer = dir_processed + "tokenizer_%s.pkl" % name_dataset
    file_label_index = dir_processed + "label_index_%s.npy" % name_dataset
    if use_loaded:
        X, y, emb = pickle.load(open(file_processed_data, "rb"))
        tokenizer = tokenizer_from_json(open(file_tokenizer, "r", encoding="utf-8").read())
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load(file_label_index)
        return X, y, emb, tokenizer, label_encoder

    cleaned_text = data["text"].apply(clean_text).values
    tokenizer = Tokenizer(num_words=max_num_words, oov_token='oov_token_placeholder')
    tokenizer.fit_on_texts(list(cleaned_text))
    tokenizer_json = tokenizer.to_json(ensure_ascii=False)
    with open(file_tokenizer, 'w', encoding='utf-8') as fout:
        fout.write(tokenizer_json)

    sequences = tokenizer.texts_to_sequences(cleaned_text)
    X = pad_sequences(sequences, maxlen=max_len_seq)
    word_index = tokenizer.word_index
    num_words = len(word_index)
    print('Found %s Words' % num_words)

    print(set(data["label"].values))
    label_encoder = LabelEncoder().fit(data["label"].values)
    np.save(file_label_index, label_encoder.classes_)
    print('Found %s Classes' % len(label_encoder.classes_))
    y = label_encoder.transform(data["label"].values)

    print('Loading Word Embeddings...')
    emb = (np.random.rand(min(num_words + 1, max_num_words), emb_dim) - 0.5) * 0.1  # +1 because idx 0 is not used
    with open(file_emb, 'r', encoding='utf-8') as fin:
        for line in fin:
            tokens = line.rstrip().split(' ')
            if tokens[0] in word_index.keys() and word_index[tokens[0]] < max_num_words:
                emb[word_index[tokens[0]]] = np.asarray(tokens[1:], dtype='float32')

    pickle.dump((X, y, emb), open(file_processed_data, "wb"))
    return X, y, emb, tokenizer, label_encoder


def clean_text(text):
    text = remove_urls(text)
    text = re.sub('[^A-Za-z0-9]', ' ', text.lower())
    text = ' '.join(text.split())
    return text


def remove_urls(text):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    return (text)


def process_data(data,
                 tokenizer,
                 label_encoder,
                 max_len_seq=35):
    cleaned_text = data["text"].apply(clean_text).values
    sequences = tokenizer.texts_to_sequences(cleaned_text)
    X = pad_sequences(sequences, maxlen=max_len_seq)
    y = label_encoder.transform(data["label"].values)

    return X, y


def read_eec(tokenizer, label_encoder, max_len_seq=35):
    # run make_madlib.py first
    data = pd.read_csv(dir_processed + "madlib.csv")
    X, y = process_data(data, tokenizer, label_encoder, max_len_seq)

    idxs_male = [i for i in range(len(data["gender"].values)) if data["gender"].values[i] == "male"]
    idxs_female = [i for i in range(len(data["gender"].values)) if data["gender"].values[i] == "female"]
    idxs = (idxs_male, idxs_female)
    return X, y, idxs


def read_weights(name_dataset):
    return np.load(dir_processed + "weights_%s.npy" % name_dataset)


def get_augmentation_data(data, tokenizer, label_encoder, max_len_seq=35):
    sensitive_words, sensitive_general_words, sensitive_extra_words = get_sensitive_words()
    data_augmented = pd.DataFrame(columns=["text", "label"])
    X = data["text"].apply(clean_text).values
    y = data["label"].values
    for i in range(len(X)):
        text, text_processed = X[i].split(" "), []
        has_sensitive = False
        for word in text:
            if word in sensitive_words:
                has_sensitive = True
                text_processed.append(
                    sensitive_words[(sensitive_words.index(word) // 2) * 4 + 1 - sensitive_words.index(word)])
                continue
            text_processed.append(word)
        text_processed = " ".join(text_processed)
        if has_sensitive:
            data_augmented = data_augmented.append({"text": text_processed, "label": y[i]}, ignore_index=True)
    X, y = process_data(data_augmented, tokenizer, label_encoder, max_len_seq)
    return X, y


if __name__ == '__main__':
    data, idxs = read_data("st", False)
    preprocess_data(data, "st", False)
    tr, val, tst = idxs
