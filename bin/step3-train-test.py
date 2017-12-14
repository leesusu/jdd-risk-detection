# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, fbeta_score
from xgboost import XGBClassifier

if __name__ == '__main__':
    # 删除了timelong后的重要的特征
    imp_feats = ['lastlogin_timegap_cont', 'nexttolastlogin_timegap_cont', 'nexttolastlogin_logfrom_lld_cont', 'lastlogin_logfrom_lld_cont',
                 'nexttolastlogin_device_lld_cont', 'nexttolastlogin_result_lld_cont', 'lastlogin_ip_lld_cont', 'lastlogin_result_lld_cont', 'nexttolastlogin_city_lld_cont',
                 'nexttolastlogin_ip_lld_cont', 'lastlogin_type_lld_cont', 'nexttolastlogin_type_lld_cont', 'lastlogin_device_lld_cont', 'lastlogin_city_lld_cont']

    # 特征的log变换等
    data_train = pd.read_csv('../data/train_test/train.csv')
    data_train['lastlogin_timelong_cont'] = data_train['lastlogin_timelong_cont'].map(lambda i: np.nan if (i <= 0) or (pd.isna(i)) else np.log10(i + 1))
    data_train['lastlogin_timegap_cont'] = data_train['lastlogin_timegap_cont'].map(lambda i: np.log10(i + 1))
    data_train['nexttolastlogin_timelong_cont'] = data_train['nexttolastlogin_timelong_cont'].map(lambda i: np.nan if (i <= 0) or (pd.isna(i)) else np.log10(i + 1))
    data_train['nexttolastlogin_timegap_cont'] = data_train['nexttolastlogin_timegap_cont'].map(lambda i: np.log10(i + 1))
    for i in data_train.columns:
        if i.endswith('_lld_cont'):
            data_train[i] = data_train[i].map(lambda i: np.nan if pd.isna(i) else np.log10(i * 1000 + 1))

    # 只保留重要的特征
    data_train = data_train[['is_risk'] + imp_feats]

    # 节省内存空间
    for c, dtype in zip(data_train.columns, data_train.dtypes):
        if dtype == np.int64:
            data_train[c] = data_train[c].astype(np.int32)
        elif dtype == np.float64:
            data_train[c] = data_train[c].astype(np.float32)

    X_train = data_train.drop(['is_risk'], axis=1)
    y_train = data_train['is_risk']

    # 设置pipline
    pl = Pipeline([('clf', XGBClassifier())])
    params = {}
    # params = {'clf__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    #           'clf__learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
    #           'clf__n_estimators': [10, 20, 50, 100, 200, 500, 1000],
    #           'clf__gamma': [i / 10.0 for i in xrange(0, 5)],
    #           'clf__min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #           'clf__subsample': [i / 10.0 for i in xrange(6, 10)],
    #           'clf__colsample_bytree': [i / 10.0 for i in xrange(6, 10)],
    #           'clf__reg_alpha': [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
    #           'clf__reg_lambda': [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]}

    # n_jobs设置多线程数量，-1表示全部内核数，可以手动设置成内核数减1，小心电脑卡死
    gs = GridSearchCV(pl, params, n_jobs=-1, cv=10, verbose=2, refit=True, scoring=make_scorer(fbeta_score, beta=0.1))
    # gs = RandomizedSearchCV(pl, params, n_jobs=-1, cv=10, verbose=2, refit=True, scoring=make_scorer(fbeta_score, beta=0.1))

    # 训练
    gs.fit(X_train, y_train)

    # 特征重要性
    # feat_scores = pd.DataFrame(
    #     list(zip([i.replace(' ', '') for i in gs.best_estimator_.named_steps['clf'].get_booster().feature_names], gs.best_estimator_.named_steps['clf'].feature_importances_)),
    #     columns=['feat', 'imp']).sort_values(['imp'], ascending=False)
    # sns.barplot(x='imp', y='feat', data=feat_scores, palette="Blues_d")
    # plt.show()

    # 输出最佳参数和最佳得分，由于params参数为空，因此GridSearchCV等于没调参
    print(gs.best_params_)
    print(gs.best_score_)

    # 预测
    data_test = pd.read_csv('../data/train_test/test.csv')
    data_test['lastlogin_timelong_cont'] = data_test['lastlogin_timelong_cont'].map(lambda i: np.nan if (i <= 0) or (pd.isna(i)) else np.log10(i + 1))
    data_test['lastlogin_timegap_cont'] = data_test['lastlogin_timegap_cont'].map(lambda i: np.log10(i + 1))
    data_test['nexttolastlogin_timelong_cont'] = data_test['nexttolastlogin_timelong_cont'].map(lambda i: np.nan if (i <= 0) or (pd.isna(i)) else np.log10(i + 1))
    data_test['nexttolastlogin_timegap_cont'] = data_test['nexttolastlogin_timegap_cont'].map(lambda i: np.log10(i + 1))
    for i in data_test.columns:
        if i.endswith('_lld_cont'):
            data_test[i] = data_test[i].map(lambda i: np.nan if pd.isna(i) else np.log10(i * 1000 + 1))

    # 只保留重要的特征
    data_test = data_test[['rowkey'] + imp_feats]

    # 节省内存空间
    for c, dtype in zip(data_test.columns, data_test.dtypes):
        if dtype == np.int64:
            data_test[c] = data_test[c].astype(np.int32)
        elif dtype == np.float64:
            data_test[c] = data_test[c].astype(np.float32)

    X_test = data_test.drop(['rowkey'], axis=1)
    rowkey_test = data_test['rowkey']

    # 输出submit结果，0.7的分类阈值见微信文章介绍
    y_test = pd.Series(gs.predict_proba(X_test).transpose()[1]).map(lambda i: 1 if i > 0.7 else 0)
    pd.concat([rowkey_test, y_test], axis=1).to_csv('../data/train_test/submit.csv', header=False, index=False)
