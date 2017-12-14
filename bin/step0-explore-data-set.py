# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', 180)

if __name__ == '__main__':
    # 读取登录信息
    df_login_part1 = pd.read_csv('../data/original/t_login.csv')
    df_login_part2 = pd.read_csv('../data/original/t_login_test.csv')
    df_login = pd.concat([df_login_part1, df_login_part2], ignore_index=True)

    # 读取交易信息
    df_trade_part1 = pd.read_csv('../data/original/t_trade.csv')
    df_trade_part2 = pd.read_csv('../data/original/t_trade_test.csv')
    df_trade_part2['is_risk'] = '???'
    df_trade = pd.concat([df_trade_part1, df_trade_part2], ignore_index=True)

    # 合并login和trade
    df_all = pd.concat([df_login, df_trade], ignore_index=True)
    df_all = df_all.fillna('')

    # 登录次数统计
    df_login['month'] = pd.Series(map(lambda x: x[5:7], df_login['time']))
    plt.subplot(2, 3, 1)
    sns.countplot(y=df_login['month'], palette='rainbow')
    plt.title('login count')

    # 交易次数统计
    df_trade['month'] = pd.Series(map(lambda x: x[5:7], df_trade['time']))
    plt.subplot(2, 3, 4)
    sns.countplot(y=df_trade['month'], palette='rainbow')
    plt.title('trade count')

    # is_risk全体统计
    plt.subplot(2, 3, 2)
    sns.countplot(y=df_trade_part1['is_risk'], palette='rainbow')
    plt.title('is_risk count')

    # is_risk按月统计
    df_trade_part1['month'] = df_trade_part1['time'].map(lambda x: x[5:7])
    df_trade_part1['hour'] = df_trade_part1['time'].map(lambda x: x[11:13])
    plt.subplot(2, 3, 5)
    sns.countplot(y='month', hue='is_risk', data=df_trade_part1, palette="rainbow")
    plt.title('monthly is_risk count')

    # is_risk按时间段统计
    plt.subplot(2, 3, 3)
    sns.countplot(x='hour', data=df_trade_part1[df_trade_part1['is_risk'] == 0], palette='Blues')
    plt.title('is_risk=0')
    plt.subplot(2, 3, 6)
    sns.countplot(x='hour', data=df_trade_part1[df_trade_part1['is_risk'] == 1], palette='Blues')
    plt.title('is_risk=1')

    plt.show()

    # 全部数据按人groupby，并按时间排序
    for id, df_this_id in df_all.groupby('id'):
        df_this_id = df_this_id.sort_values('time')

        # 只查看 7月有交易行为 且 曾经有风险交易 的人的日志
        if ('???' in df_this_id['is_risk'].values) and (1 in df_this_id['is_risk'].values):
            print(df_this_id)
            print('----------------------')
