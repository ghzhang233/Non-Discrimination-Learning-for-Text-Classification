from collections import defaultdict
import math
import numpy as np

from data_utils import read_data, preprocess_data, clean_text, read_weights

if __name__ == '__main__':
    data, _ = read_data(use_loaded=True)
    X, y, emb, tokenizer, label_encoder = preprocess_data(data=data, use_loaded=True)
    with open("data/adjectives_people.txt", "r", encoding="utf-8") as fin:
        identity_columns = [line.strip() for line in fin.readlines()]

    cleaned_text = data["text"].apply(clean_text).values

    debias_weights = np.ones(len(y))
    num_pos, num_all = defaultdict(int), defaultdict(int)
    sum_pos = sum([debias_weights[i] for i in range(len(y)) if y[i] == 0])
    sum_all = sum(debias_weights)
    for idty in identity_columns:
        for i in range(len(cleaned_text)):
            ok = False
            sen = cleaned_text[i]
            if idty in ["american", "african"]:
                sen_split = sen.split()
                for j in range(len(sen_split)):
                    if sen_split[j] == idty:
                        if j == 0 or " ".join([sen_split[j - 1], sen_split[j]]) != "american african":
                            if j == len(sen_split) - 1 or " ".join(
                                    [sen_split[j], sen_split[j + 1]]) != "american african":
                                ok = True
                                break
            elif (len(idty.split()) == 1 and idty in sen.split()) or (len(idty.split()) == 2 and sen.find(idty) != -1):
                ok = True
            else:
                ok = False
            if ok:
                num_all[idty] += debias_weights[i]
                if y[i] == 0:
                    num_pos[idty] += debias_weights[i]
    ret = []
    for idty in identity_columns:
        num_pos[idty] /= sum_pos
        num_all[idty] /= sum_all
        ret.append([idty, num_pos[idty] * 100, num_all[idty] * 100, num_pos[idty] * 100 - num_all[idty] * 100])

    print("-----------------------------------------------")

    debias_weights = read_weights()
    num_pos, num_all = defaultdict(int), defaultdict(int)
    sum_pos = sum([debias_weights[i] for i in range(len(y)) if y[i] == 0])
    sum_all = sum(debias_weights)
    for idty in identity_columns:
        for i in range(len(cleaned_text)):
            ok = False
            sen = cleaned_text[i]
            if idty in ["american", "african"]:
                sen_split = sen.split()
                for j in range(len(sen_split)):
                    if sen_split[j] == idty:
                        if j == 0 or " ".join([sen_split[j - 1], sen_split[j]]) != "american african":
                            if j == len(sen_split) - 1 or " ".join(
                                    [sen_split[j], sen_split[j + 1]]) != "american african":
                                ok = True
                                break
            elif (len(idty.split()) == 1 and idty in sen.split()) or (len(idty.split()) == 2 and sen.find(idty) != -1):
                ok = True
            else:
                ok = False
            if ok:
                num_all[idty] += debias_weights[i]
                if y[i] == 0:
                    num_pos[idty] += debias_weights[i]

    for i, idty in enumerate(identity_columns):
        num_pos[idty] /= sum_pos
        num_all[idty] /= sum_all
        ret[i] += [num_pos[idty] * 100, num_all[idty] * 100, num_pos[idty] * 100 - num_all[idty] * 100]
    ret = sorted(ret, key=lambda x: -math.fabs(x[3]))
    for i in range(len(ret)):
        print("%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%s" %
              (ret[i][1], ret[i][2], ret[i][3], ret[i][4], ret[i][5], ret[i][6], ret[i][0]))
