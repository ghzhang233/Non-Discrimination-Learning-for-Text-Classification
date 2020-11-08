import pandas as pd
from scipy.stats import ttest_ind


def significance_test(v1, v2):
    _, p_val = ttest_ind(v1, v2)

    if p_val <= 0.01:
        print('\tSignificant at 0.01, p-value: %.4f' % p_val)
    elif p_val <= 0.05:
        print('\tSignificant at 0.05, p-value: %.4f' % p_val)
    else:
        print('\tNot significant,     p-value: %.4f' % p_val)

normal = pd.read_csv("results/results_summary_biased.csv")
weight = pd.read_csv("results/results_summary_weight.csv")

for i in ["auc_test", "auc_eec", "efpr_eec", "efnr_eec"]:
    print(i.upper(), end="\n")
    print("Biased:Weighted:\t%.3lf\t%.3lf" % (normal[i].values.mean(), weight[i].values.mean()), end="")
    significance_test(normal[i].values, weight[i].values)
