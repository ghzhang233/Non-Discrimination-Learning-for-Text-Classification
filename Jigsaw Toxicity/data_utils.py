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

def txtload(filename):
    ret = []
    with open(filename, "r") as fin:
        for line in fin.readlines():
            ret.append(int(line.strip()))
    return np.array(ret)

def read_data(use_loaded=True):
    file_data = dir_processed + "data_origin.pkl"
    if use_loaded:
        data, idxs = pickle.load(open(file_data, "rb"))
        return data, idxs

    # read data
    data = pd.read_csv("data/train.csv").fillna(0)
    data = data.rename(columns={"comment_text": "text"})
    data["label"] = data["target"].apply(lambda x: "BAD" if x >= 0.5 else "NOT_BAD")

    # data = data.rename(cols={""})

    num_samples = len(data)
    orders = np.random.permutation(num_samples)
    idxs_train = txtload("data/train_idx.txt")
    idxs_dev = txtload("data/dev_idx.txt")
    idxs_test = txtload("data/test_idx.txt")
    # idxs_train = orders[0:int(num_samples * 0.8)]
    # idxs_dev = orders[int(num_samples * 0.8): int(num_samples * 0.9)]
    # idxs_test = orders[int(num_samples * 0.9):]
    idxs = (idxs_train, idxs_dev, idxs_test)

    pickle.dump((data, idxs), open(file_data, "wb"))
    return data, idxs


def preprocess_data(data,
                    use_loaded=True,
                    file_emb="../glove.840B.300d.txt",
                    max_num_words=50000,
                    max_len_seq=35,
                    emb_dim=300):
    # preprocess data
    file_processed_data = dir_processed + "data_processed.pkl"
    file_tokenizer = dir_processed + "tokenizer.pkl"
    file_label_index = dir_processed + "label_index.npy"
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


def read_official_test():
    # read data
    data = pd.read_csv("data/test.csv")
    data = data.rename(columns={"comment_text": "text"})
    return data


def process_data(data,
                 tokenizer,
                 label_encoder,
                 max_len_seq=35):
    cleaned_text = data["text"].apply(clean_text).values
    sequences = tokenizer.texts_to_sequences(cleaned_text)
    X = pad_sequences(sequences, maxlen=max_len_seq)
    y = label_encoder.transform(data["label"].values) if "label" in data.columns else None

    return X, y


def read_eec(tokenizer, label_encoder, max_len_seq=35):
    # run make_madlib.py first
    data = pd.read_csv("data/bias_madlibs_77k.csv")
    data = data.rename(columns={"Label": "label", "Text": "text"})
    X, y = process_data(data, tokenizer, label_encoder, max_len_seq)
    with open("data/adjectives_people.txt", "r", encoding="utf-8") as fin:
        identity_terms = [line.strip() for line in fin.readlines()]

    def recognize_identity(text):
        if "african american" in text:
            return "african american"
        for i in identity_terms:
            if i in text.split():
                return i
        for i in identity_terms:
            if text.find(i) != -1:
                return i
        return "null"

    data["identity"] = data["text"].apply(recognize_identity)
    idxs = dict()
    for identity in identity_terms:
        idxs[identity] = [i for i in range(len(data["identity"].values)) if data["identity"].values[i] == identity]
    return X, y, idxs


def find_idty(idty, sen):
    freq = 0
    sen_split = sen.split()

    if idty in ["american", "african"]:
        for j in range(len(sen_split)):
            if sen_split[j] == idty:
                if j == len(sen_split) - 1 or " ".join([sen_split[j], sen_split[j + 1]]) != "american african":
                    if j == 0 or " ".join([sen_split[j - 1], sen_split[j]]) != "american african":
                        freq += 1
    elif len(idty.split()) == 1:
        for j in sen_split:
            if j == idty:
                freq += 1
    elif len(idty.split()) == 2:
        freq = len(re.compile(idty).findall(sen))
    else:
        freq = 0

    return freq


def get_supplementation(data, suffix="", use_loaded=True):
    file_supplement = dir_processed + "data_supplement_%s.pkl" % suffix
    if use_loaded:
        data_ret = pickle.load(open(file_supplement, "rb"))
        return data_ret

    with open("data/adjectives_people.txt", "r", encoding="utf-8") as fin:
        identity_columns = [line.strip() for line in fin.readlines()]

    data_sup = pd.read_csv("data/wiki_debias_%s.csv" % suffix)
    data_sup = data_sup.rename(columns={"comment": "text"})
    data_sup["label"] = data_sup["is_toxic"].apply(lambda x: "BAD" if x >= 0.5 else "NOT_BAD")
    idty_sen_bad_sup_dict = defaultdict(list)
    idty_sen_good_sup_dict = defaultdict(list)
    cleaned_text_sup = data_sup["text"].apply(clean_text).values
    for idty in identity_columns:
        for i in range(len(cleaned_text_sup)):
            freq = find_idty(idty, cleaned_text_sup[i])
            if freq != 0:
                if data_sup["label"].values[i] == "BAD":
                    idty_sen_bad_sup_dict[idty].append((i, freq))
                if data_sup["label"].values[i] == "NOT_BAD":
                    idty_sen_good_sup_dict[idty].append((i, freq))

    idty_bad_freq_dict = defaultdict(int)
    idty_freq_dict = defaultdict(int)
    cleaned_text = data["text"].apply(clean_text).values
    for idty in identity_columns:
        for i in range(len(cleaned_text)):
            freq = find_idty(idty, cleaned_text[i])
            if data["label"].values[i] == "BAD":
                idty_bad_freq_dict[idty] += freq
            idty_freq_dict[idty] += freq
    p_bad = len([i for i in data["label"].values if i == "BAD"]) / len(data)

    data_ret = pd.DataFrame(columns=["text", "label"])
    for idty in identity_columns:
        if idty_freq_dict[idty] == 0:
            continue
        p_bad_idty = p_bad_idty_origin = idty_bad_freq_dict[idty] / idty_freq_dict[idty]
        if abs(p_bad_idty_origin - p_bad) <= 0.01:
            continue
        elif p_bad_idty_origin > p_bad:
            for i in idty_sen_good_sup_dict[idty]:
                text = data_sup["text"].values[i[0]]
                label = data_sup["label"].values[i[0]]
                data_ret = data_ret.append({"text": text, "label": label}, ignore_index=True)
                idty_freq_dict[idty] += 1
                p_bad_idty = idty_bad_freq_dict[idty] / idty_freq_dict[idty]
                if p_bad_idty < p_bad or abs(p_bad_idty - p_bad) <= 0.01:
                    break
        else:
            for i in idty_sen_bad_sup_dict[idty]:
                text = data_sup["text"].values[i[0]]
                label = data_sup["label"].values[i[0]]
                data_ret = data_ret.append({"text": text, "label": label}, ignore_index=True)
                idty_bad_freq_dict[idty] += 1
                idty_freq_dict[idty] += 1
                p_bad_idty = idty_bad_freq_dict[idty] / idty_freq_dict[idty]
                if p_bad_idty > p_bad or abs(p_bad_idty - p_bad) <= 0.01:
                    break
        print("%.2lf\t%.2lf\t%s" % (p_bad_idty, p_bad, idty))

    pickle.dump(data_ret, open(file_supplement, "wb"))
    return data_ret


def read_weights():
    return np.load(dir_processed + "weights.npy")


def main():
    data, idxs = read_data(False)
    preprocess_data(data, False)
    idxs_train, idxs_dev, idxs_test = idxs
    get_supplementation(data.iloc[idxs_train], "train", False)
    get_supplementation(data.iloc[idxs_dev], "dev", False)


if __name__ == '__main__':
    main()
