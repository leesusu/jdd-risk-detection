# coding=utf-8

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mssno

if __name__ == '__main__':
    data_train = pd.read_csv('../data/train_test/train.csv')

    # 特征的log变换等
    data_train['lastlogin_timelong_cont'] = data_train['lastlogin_timelong_cont'].map(lambda i: np.nan if (i <= 0) or (pd.isna(i)) else np.log10(i + 1))
    data_train['lastlogin_timegap_cont'] = data_train['lastlogin_timegap_cont'].map(lambda i: np.log10(i + 1))
    data_train['nexttolastlogin_timelong_cont'] = data_train['nexttolastlogin_timelong_cont'].map(lambda i: np.nan if (i <= 0) or (pd.isna(i)) else np.log10(i + 1))
    data_train['nexttolastlogin_timegap_cont'] = data_train['nexttolastlogin_timegap_cont'].map(lambda i: np.log10(i + 1))
    for i in data_train.columns:
        if i.endswith('_lld_cont'):
            data_train[i] = data_train[i].map(lambda i: np.nan if pd.isna(i) else np.log10(i * 1000 + 1))

    # 缺失值分析
    # mssno.matrix(data_train)
    # mssno.bar(data_train, fontsize=12)
    # plt.show()

    # 特征相关性分析
    # cor = data_train.corr()
    # mask = np.zeros_like(cor)
    # indices = np.triu_indices_from(cor)
    # mask[indices] = True
    # sns.heatmap(cor, mask=mask, vmin=-1, vmax=1, linewidths=0.5, cmap=sns.diverging_palette(220, 8, as_cmap=True), annot=True)
    # plt.show()

    X_pos = data_train[data_train['is_risk'] == 1].drop(['is_risk'], axis=1)
    X_pos = X_pos[[i for i in X_pos.columns if i.startswith('lastlogin_')]]
    X_neg = data_train[data_train['is_risk'] == 0].drop(['is_risk'], axis=1)
    X_neg = X_neg[[i for i in X_neg.columns if i.startswith('lastlogin_')]]
    feat_cnt = X_pos.shape[1]
    plt.figure(figsize=[10, 3 * feat_cnt])
    plt.subplots_adjust(hspace=0.4)
    for i, feat in enumerate(X_pos.columns):
        plt.subplot(feat_cnt, 1, i + 1)
        plt.title('histogram of feature: ' + feat)
        plt.grid(True, linestyle='--')

        x_pos = pd.Series([x for x in X_pos[feat] if not math.isnan(x)])
        x_neg = pd.Series([x for x in X_neg[feat] if not math.isnan(x)])
        sns.distplot(x_pos, kde=x_pos.value_counts().shape[0] > 1)
        sns.distplot(x_neg, kde=x_neg.value_counts().shape[0] > 1)
    plt.savefig('../data/train_test/view_features.png', format='png', dpi=300, bbox_inches='tight')
