# coding=utf-8

import time
import numpy as np
import pandas as pd


def get_trade_with_login():
    data_res = []
    for i in df_trade.to_dict(orient='records'):
        if i['id'] in data_login:
            df_login_this_id = data_login[i['id']]
            df = df_login_this_id[df_login_this_id['time'] < i['time']]

            if df.empty:
                ele = [''] * 12
            elif df.shape[0] == 1:
                ele = df[['log_from', 'device', 'ip', 'city', 'type', 'result']].iloc[-1].tolist() + [''] * 6
            else:
                ele = df[['log_from', 'device', 'ip', 'city', 'type', 'result']].iloc[-1].tolist() + df[['log_from', 'device', 'ip', 'city', 'type', 'result']].iloc[-2].tolist()

            data_res.append(ele)
        else:
            data_res.append([''] * 12)

    df_res = pd.DataFrame(data_res, columns=magic_cols)
    return pd.concat([df_trade, df_res], axis=1)


def get_feats(df_trade_part):
    data_res = []
    for i in df_trade_part.to_dict(orient='records'):
        trade_time = int(time.mktime(time.strptime(i['time'], '%Y-%m-%d %H:%M:%S')))

        if i['id'] in data_login:
            df_login_this_id = data_login[i['id']]
            df = df_login_this_id[df_login_this_id['time'] < i['time']]

            if df.empty:
                ele = [''] * 4
            elif df.shape[0] == 1:
                ele = [df['timelong'].values[-1], trade_time - df['timestamp'].values[-1]] + [''] * 2
            else:
                ele = [df['timelong'].values[-1], trade_time - df['timestamp'].values[-1], df['timelong'].values[-2], trade_time - df['timestamp'].values[-2]]

            data_res.append(ele)
        else:
            data_res.append([''] * 4)
    return pd.DataFrame(data_res, columns=['lastlogin_timelong_cont', 'lastlogin_timegap_cont', 'nexttolastlogin_timelong_cont', 'nexttolastlogin_timegap_cont'])


if __name__ == '__main__':
    output_lld_features = '../data/train_test/lld_features.csv'
    output_train = '../data/train_test/train.csv'
    output_test = '../data/train_test/test.csv'

    # 读取登录数据
    dtype_login = {'log_id': str, 'timelong': np.int32, 'device': str, 'log_from': str, 'ip': str, 'city': str, 'result': str, 'timestamp': np.int32, 'type': str, 'id': str,
                   'is_scan': np.int32, 'is_sec': np.int32, 'time': str}
    df_login_part1 = pd.read_csv('../data/original/t_login.csv', dtype=dtype_login)
    df_login_part2 = pd.read_csv('../data/original/t_login_test.csv', dtype=dtype_login)
    df_login = pd.concat([df_login_part1, df_login_part2], ignore_index=True)
    df_login['timelong'] = list(map(lambda x, y: x * 1000 if y in ['10', '11'] else x, df_login['timelong'], df_login['log_from']))  # log_from为10或11时，timelong的单位是秒，其他情况为毫秒

    # 读取交易数据
    dtype_trade = {'rowkey': str, 'time': str, 'id': str, 'is_risk': np.int32}
    df_trade_part1 = pd.read_csv('../data/original/t_trade.csv', dtype=dtype_trade)
    df_trade_part1['time'] = df_trade_part1['time'].map(lambda x: x[:19])  # trade里的time都以“.0”结尾，删之
    df_trade_part2 = pd.read_csv('../data/original/t_trade_test.csv', dtype=dtype_trade)
    df_trade_part2['time'] = df_trade_part2['time'].map(lambda x: x[:19])
    df_trade = pd.concat([df_trade_part1, df_trade_part2], ignore_index=True)

    # 登录信息按id groupby、排序，并放到dict里，方便按id索引
    data_login = {}
    for id, df in df_login.groupby(['id']):
        data_login[id] = df.sort_values(['time'])

    # 交易信息按id groupby、排序，并放到dict里，方便按id索引
    # data_trade = {}
    # for id, df in df_trade.groupby(['id']):
    #     data_trade[id] = df.sort_values(['time'])

    # 关联了登录信息的新的列名
    magic_cols = ['lastlogin_logfrom', 'lastlogin_device', 'lastlogin_ip', 'lastlogin_city', 'lastlogin_type', 'lastlogin_result',
                  'nexttolastlogin_logfrom', 'nexttolastlogin_device', 'nexttolastlogin_ip', 'nexttolastlogin_city', 'nexttolastlogin_type', 'nexttolastlogin_result']

    # 关联每次交易的上次登录、上上次登录信息，并按time排序
    df_trade_with_login = get_trade_with_login()
    df_trade_with_login = df_trade_with_login.sort_values(['time'])

    # 生成类似likelihood encoding的异常程度的特征
    magic_dict = {}
    data_res = []
    for i in df_trade_with_login.to_dict(orient='records'):
        ele = [i['rowkey'], i['time'], i['id']]
        for j in magic_cols:
            prefix = j + '\t' + i[j]
            key_sum = prefix + '\tsum'
            key_len = prefix + '\tlen'

            if key_sum not in magic_dict:
                magic_dict[key_sum] = 0.0
            if key_len not in magic_dict:
                magic_dict[key_len] = 0.0

            if magic_dict[key_len] < 5.0:
                ele.append('')
            else:
                ele.append(str(magic_dict[key_sum] / magic_dict[key_len]))

            if i['is_risk'] == 1:
                magic_dict[key_sum] += 1.0
                magic_dict[key_len] += 1.0
            elif i['is_risk'] == 0:
                magic_dict[key_len] += 1.0

        data_res.append(ele)

    df_lld_feats = pd.DataFrame(data_res, columns=['rowkey', 'time', 'id'] + [i + '_lld_cont' for i in magic_cols])
    df_lld_feats.to_csv(output_lld_features, index=False)

    # 生成训练集
    df_train = pd.merge(df_trade_part1, df_lld_feats, on=['rowkey', 'time', 'id'])
    df_train = pd.concat([df_train, get_feats(df_trade_part1)], axis=1)
    df_train = df_train[df_train['time'] > '2015-01-31 23:59:59']  # 去除1月的trade数据
    df_train = df_train[[i for i in df_train.columns if i not in ['rowkey', 'time', 'id']]]  # 删去无用字段，只保留is_risk和features
    df_train.to_csv(output_train, index=False)

    # 生成测试集
    df_test = pd.merge(df_trade_part2, df_lld_feats, on=['rowkey', 'time', 'id'])
    df_test = pd.concat([df_test, get_feats(df_trade_part2)], axis=1)
    df_test = df_test[[i for i in df_test.columns if i not in ['time', 'id']]]  # 删去无用字段，只保留rowkey和features
    df_test.to_csv(output_test, index=False)
