import pandas as pd
import numpy as np
import os
import datetime as dt
from datetime import timedelta
from optparse import OptionParser
import warnings
from multiprocessing import Pool
warnings.filterwarnings('ignore')

substra_name = 'USDT_SPREAD'
exchanges = ['binance', 'okex', 'ftx', 'bybit', 'gateio', 'huobi', 'bitfinex']


def create_dataframe(path, exchange):
    raw_ = pd.read_pickle(path)
    raw_data = raw_[exchange == raw_["exchange"]]
    raw_data.last_price = raw_data.last_price.astype(float)
    raw_data.pair = raw_data.pair.astype(str)
    raw_data.exchange = raw_data.exchange.astype(str)
    raw_data["last_price"].fillna(0, inplace=True)
    raw = raw_data[['pair','last_price', 'end_date', 'exchange']]
    rawdata = raw.pivot_table(index="end_date", columns="pair", values="last_price")
    rawdata = rawdata[~rawdata.index.duplicated(keep='first')]
    rawdata = (rawdata.ffill()+rawdata.bfill())/2
    rawdata.index = rawdata.index+timedelta(seconds=1)
    selected = []
    for col in rawdata.columns:
        if 'USDT' in col:
            selected.append(col)
    rawdata = rawdata[selected]
    return rawdata


def create_new_price(path, exchange):
    rawdata = pd.read_csv(path, index_col=0)
    rawdata = rawdata[rawdata.exchange==exchange]
    rawdata = rawdata[['last','enddate','symbol']]
    fut = rawdata.pivot_table(values='last', index='enddate', columns='symbol')
    fut.index = pd.to_datetime(pd.Series(pd.to_datetime(fut.index).astype(str)).apply(lambda x: x[:-6]))+timedelta(seconds=1)
    selected = []
    for col in fut.columns:
        if 'USDT' in col:
            selected.append(col)
    fut = fut[selected]
    for col in fut.columns:
        fut.rename(columns={col:col.split('/')[0]}, inplace=True)
    return fut


def create_new_rate(path, exchange):
    rawdata = pd.read_csv(path, index_col=0)
    rawdata = rawdata[rawdata.exchange==exchange]
    rawdata = rawdata[['funding_rate','enddate','symbol']]
    fut = rawdata.pivot_table(values='funding_rate', index='enddate', columns='symbol')
    fut.index = pd.to_datetime(pd.Series(pd.to_datetime(fut.index).astype(str)).apply(lambda x: x[:-6]))+timedelta(seconds=1)
    selected = []
    for col in fut.columns:
        if 'USDT' in col:
            selected.append(col)
    fut = fut[selected]
    for col in fut.columns:
        fut.rename(columns={col:col.split('/')[0]}, inplace=True)
    return fut


class Path():
    def __init__(self, type='last_price', env='TDS'):
        self.type = type
        self.env = env
        if 'TDS' == self.env:
            using_path='tardis_data'
        else:
            using_path='mix_data'
        self.path_dict = {}
        for exchange in exchanges:
            self.path_dict[exchange] = '/home/{}/futures/usdt/hour_twap/{}/{}.pkl'.format(using_path, self.type, exchange)

class Data():
    def __init__(self, type='last_price', env='TDS'):
        self.type = type
        self.env = env
        self.path_dict = Path(self.type,self.env).path_dict
        self.data_dict = {}
        for exchange in exchanges:
            self.data_dict[exchange] = pd.read_pickle(self.path_dict[exchange])


class Sub_generator():
    def __init__(self, env='TDS', days=3):
        self.days=days
        self.exchanges = ['binance','okex','ftx','bybit','gateio','huobi','bitfinex']
        self.env = env
        self.price_data = Data('last_price', env=env).data_dict
        self.funding_data = Data('funding_rate', env=env).data_dict
    def non_ftx_sub(self,short,long):
            if short=='ftx' or long=='ftx':
                raise 'ftx in non ftx func'
            if self.env == 'TDS':
                out_folder = '/home/strategy/usdt_spread/'
            else:
                out_folder = '/home/pro_strategy/usdt_spread/'

            coinnum_path = out_folder + 'coinnum/' + '{}2{}'.format(short,long) + '_coinnum.pkl'
            margin_path = out_folder + 'margin/' + '{}2{}'.format(short,long) + '_margin.pkl'
            spread_path = out_folder + 'spread/' + '{}2{}'.format(short,long) + '_spread.pkl'
            total_path = out_folder + 'total/' + '{}2{}'.format(short,long) + '_total.pkl'

            short_fut = self.price_data[short].T.dropna(how='all').T.loc['2021-06-01':]
            short_rate = self.funding_data[short].T.dropna(how='all').T.loc['2021-06-01':]

            long_fut = self.price_data[long].T.dropna(how='all').T.loc['2021-06-01':]
            long_rate = self.funding_data[long].T.dropna(how='all').T.loc['2021-06-01':]

            if self.env == 'MIX':
                live_rate_path = '/home/raw_data/funding/live_data/'
                live_price_path = '/home/raw_data/futures/live_data/'
                live_rate_file_path = os.path.join(live_rate_path, np.sort(os.listdir(live_rate_path))[-1])
                live_price_file_path = os.path.join(live_price_path, np.sort(os.listdir(live_price_path))[-1])
                short_new_rate = create_new_rate(live_rate_file_path, short)
                short_new_fut = create_new_price(live_price_file_path, short)
                for col in short_new_fut.columns:
                    short_new_fut.rename(columns={col: col.split('/')[0]}, inplace=True)
                    short_new_rate.rename(columns={col: col.split('/')[0]}, inplace=True)
                short_fut = short_fut.loc[:short_new_fut.index[0]]
                short_rate = short_rate.loc[:short_new_rate.index[0]]
                short_fut = pd.concat([short_fut, short_new_fut], sort=True)
                short_rate = pd.concat([short_rate, short_new_rate], sort=True)

                long_new_rate = create_new_rate(live_rate_file_path, long)
                long_new_fut = create_new_price(live_price_file_path, long)
                long_rate = long_rate.loc[:long_new_rate.index[0]]
                long_fut = long_fut.loc[:long_new_fut.index[0]]
                long_rate = pd.concat([long_rate, long_new_rate], sort=True)
                long_fut = pd.concat([long_fut, long_new_fut], sort=True)

            end_date = min(long_fut.index[-1], long_rate.index[-1], short_fut.index[-1], short_rate.index[-1])
            short_fut = short_fut.loc[:end_date + timedelta(seconds=1)]
            short_rate = short_rate.loc[:end_date + timedelta(seconds=1)]
            long_fut = long_fut.loc[:end_date + timedelta(seconds=1)]
            long_rate = long_rate.loc[:end_date + timedelta(seconds=1)]

            col_names = np.sort(list(
                set(long_fut.columns) & set(short_fut.columns) & set(short_rate.columns) & set(long_rate.columns)))
            long_rate = long_rate[col_names]
            long_fut = long_fut[col_names]
            short_fut = short_fut[col_names]
            short_rate = short_rate[col_names]
            if len(short_rate) != len(long_fut):
                print('DataProblem Warning: this will execute with intersection index')
                slt_idx = np.sort(list(set(long_rate.index)&set(long_fut.index)&set(short_fut.index)&set(short_rate.index)))
                long_rate = long_rate.loc[slt_idx]
                long_fut = long_fut.loc[slt_idx]
                short_fut = short_fut.loc[slt_idx]
                short_rate = short_rate.loc[slt_idx]

            if not (os.path.exists(coinnum_path) and os.path.exists(margin_path) and os.path.exists(
                    spread_path) and os.path.exists(total_path)):
                margin_l = []
                coinnum_l = []
                print('generate new exe pair table')
                i=0
                used_cols =[]
                for coin_name in col_names:
                    try:
                        start_date = max(long_fut[coin_name].dropna().index[0], short_fut[coin_name].dropna().index[0],
                                         long_rate[coin_name].dropna().index[0], short_rate[coin_name].dropna().index[0])
                        standard_index = long_fut[coin_name].loc[start_date:].replace(0,np.nan).dropna().index.to_list()
                        margin_common = pd.Series(index=standard_index, dtype=float)
                        margin_common.iloc[0] = 0.5
                        realized_pnl = pd.Series(index=standard_index, dtype=float)
                        funding = pd.Series(index=standard_index, dtype=float)
                        trade_cost = pd.Series(index=standard_index, dtype=float)
                        coin_num = pd.DataFrame(index=standard_index, columns=[coin_name])
                        coin_num.iloc[0] = 1 / ((short_fut[coin_name]).replace(0, np.nan).dropna().iloc[0])
                        binance_holdprice = pd.DataFrame(index=standard_index, columns=[coin_name])
                        okex_holdprice = pd.DataFrame(index=standard_index, columns=[coin_name])
                        binance_holdprice.iloc[0] = short_fut.loc[standard_index[1], coin_name]
                        okex_holdprice.iloc[0] = long_fut.loc[standard_index[1], coin_name]

                    except:
                        continue
                    used_cols.append(coin_name)
                    idx = 0
                    for mi in standard_index[1:]:

                        idx += 1
                        last_mi = standard_index[idx - 1]

                        if mi.hour % 8 != 0:
                            margin_common.loc[mi] = margin_common.loc[last_mi]
                            funding.loc[mi] = 0
                        else:
                            funding_from_okex = ((-long_rate.loc[mi, coin_name]) * (coin_num.loc[last_mi]) * (
                            long_fut.loc[mi, coin_name])).sum()
                            funding_from_binance = ((short_rate.loc[mi, coin_name]) * (coin_num.loc[last_mi]) * (
                            short_fut.loc[mi, coin_name])).sum()
                            margin_common.loc[mi] = margin_common.loc[
                                                        last_mi] + funding_from_okex + funding_from_binance
                            funding.loc[mi] = funding_from_okex + funding_from_binance
                        future_pnl = coin_num.loc[last_mi] * (
                                    long_fut.loc[mi, coin_name] - (long_fut.loc[last_mi, coin_name]) + short_fut.loc[
                                last_mi, coin_name] - short_fut.loc[mi, coin_name])
                        realized_pnl.loc[mi] = future_pnl.values
                        margin_common.loc[mi] = np.float64(margin_common.loc[mi] + future_pnl)
                        total_basket_value = coin_num.loc[last_mi] * (
                                    short_fut.loc[mi, coin_name] + long_fut.loc[mi, coin_name])
                        upper_limit = total_basket_value * 0.275
                        lower_limit = total_basket_value * 0.20

                        if margin_common.loc[mi] > upper_limit.values:
                            enlarge_amount = ((margin_common.loc[mi] / 0.25) - total_basket_value)
                            enlarge_num = enlarge_amount / (
                                        short_fut.loc[mi, coin_name] + long_fut.loc[mi, coin_name])
                            coin_num.loc[mi] = (coin_num.loc[last_mi].values + enlarge_num.values)

                        elif margin_common.loc[mi] < lower_limit.values:
                            shrink_amount = (total_basket_value - (margin_common.loc[mi] / 0.25))
                            shrink_num = shrink_amount / (short_fut.loc[mi, coin_name] + long_fut.loc[mi, coin_name])
                            coin_num.loc[mi] = np.float64(coin_num.loc[last_mi] - shrink_num)


                        else:
                            coin_num.loc[mi] = coin_num.loc[last_mi]
                            okex_holdprice.loc[mi] = long_fut.loc[last_mi, coin_name]
                            binance_holdprice.loc[mi] = short_fut.loc[last_mi, coin_name]

                        okex_transaction_fee = np.sum(
                            abs((coin_num.loc[mi]).sub(coin_num.loc[last_mi], fill_value=0)) * long_fut.loc[
                                mi, coin_name] * 0.0007)
                        binance_transaction_fee = np.sum(
                            abs((coin_num.loc[mi]).sub(coin_num.loc[last_mi], fill_value=0)) * long_fut.loc[
                                mi, coin_name] * 0.0007)
                        trade_cost.loc[mi] = (binance_transaction_fee + okex_transaction_fee)
                        margin_common.loc[mi] = margin_common.loc[mi] - binance_transaction_fee - okex_transaction_fee
                        if (margin_common.loc[mi] < 0 or margin_common.loc[mi] / margin_common.loc[last_mi] > 1.1 or
                                margin_common.loc[mi] / margin_common.loc[last_mi] < 0.9)or np.isnan(margin_common.loc[mi]):
                            margin_common.loc[mi] = margin_common.loc[last_mi]
                            realized_pnl = pd.Series(index=standard_index, dtype=float)
                            coin_num.loc[mi] = coin_num.loc[last_mi]
                    i+=1
                    if i%30 == 0:
                        print('Progress:{} %'.format(round(i/len(col_names),3)*100))
                    margin_l.append(margin_common)
                    coinnum_l.append(coin_num)
                print('new table generation finished')
                long_margin = pd.concat(margin_l, axis=1)
                long_coinnum = pd.concat(coinnum_l, axis=1)
                long_margin.columns = used_cols
                long_coinnum.columns = used_cols
                long_total = long_margin + long_coinnum * (long_fut - short_fut)
                long_margin.to_pickle(margin_path)
                long_coinnum.to_pickle(coinnum_path)
                (long_total / long_total.shift(1)).cumprod().to_pickle(total_path)
                (short_fut / long_fut).to_pickle(spread_path)
            else:
                print('updating existing table')
                coinnum = pd.read_pickle(coinnum_path)
                margin = pd.read_pickle(margin_path)
                col_names = np.sort(list(set(col_names) & set(coinnum.columns)))

                long_rate = long_rate[col_names]
                long_fut = long_fut[col_names]
                short_fut = short_fut[col_names]
                short_rate = short_rate[col_names]

                coinnum.index = pd.to_datetime(coinnum.index)
                margin.index = pd.to_datetime(margin.index)
                coinnum = coinnum.reindex(short_fut.index)
                margin = margin.reindex(short_fut.index)
                margin_l = []
                coinnum_l = []

                for coin_name in col_names:

                    try:
                        start_date = max(long_fut[coin_name].dropna().index[0], short_fut[coin_name].dropna().index[0],
                                         long_rate[coin_name].dropna().index[0],
                                         short_rate[coin_name].dropna().index[0])
                        standard_index = long_fut[coin_name].loc[start_date:].replace(0,np.nan).dropna().index.to_list()

                        start_idx = standard_index.index(coinnum.dropna(how='all').index[-1])
                        margin_common = margin[coin_name]
                        realized_pnl = pd.Series(index=standard_index, dtype=float)
                        funding = pd.Series(index=standard_index, dtype=float)
                        trade_cost = pd.Series(index=standard_index, dtype=float)
                        coin_num = pd.DataFrame(coinnum[coin_name])
                        binance_holdprice = pd.DataFrame(index=standard_index, columns=[coin_name])
                        ftx_holdprice = pd.DataFrame(index=standard_index, columns=[coin_name])
                        binance_holdprice.iloc[0] = short_fut.loc[standard_index[1], coin_name]
                        ftx_holdprice.iloc[0] = long_fut.loc[standard_index[1], coin_name]
                        idx = start_idx - self.days * 24
                    except:
                        if standard_index[-1] < coinnum.dropna(how='all').index[-1]:
                            margin_common = margin[coin_name].loc[:standard_index[-1]]
                            coin_num = pd.DataFrame(coinnum[coin_name]).loc[:standard_index[-1]]
                            margin_l.append(margin_common)
                            coinnum_l.append(coin_num)
                        else:
                            margin_common = margin[coin_name]
                            coin_num = pd.DataFrame(coinnum[coin_name])
                            margin_l.append(margin_common)
                            coinnum_l.append(coin_num)
                        continue

                    for mi in standard_index[start_idx + 1 - self.days * 24:]:

                        idx += 1
                        last_mi = standard_index[idx - 1]

                        if mi.hour % 8 != 0:
                            margin_common.loc[mi] = margin_common.loc[last_mi]
                            funding.loc[mi] = 0
                        else:
                            funding_from_okex = ((-long_rate.loc[mi, coin_name]) * (coin_num.loc[last_mi]) * (
                            long_fut.loc[mi, coin_name])).sum()
                            funding_from_binance = ((short_rate.loc[mi, coin_name]) * (coin_num.loc[last_mi]) * (
                            short_fut.loc[mi, coin_name])).sum()
                            margin_common.loc[mi] = margin_common.loc[
                                                        last_mi] + funding_from_okex + funding_from_binance
                            funding.loc[mi] = funding_from_okex + funding_from_binance
                        future_pnl = coin_num.loc[last_mi] * (
                                    long_fut.loc[mi, coin_name] - (long_fut.loc[last_mi, coin_name]) + short_fut.loc[
                                last_mi, coin_name] - short_fut.loc[mi, coin_name])
                        realized_pnl.loc[mi] = future_pnl.values
                        margin_common.loc[mi] = np.float64(margin_common.loc[mi] + future_pnl)
                        total_basket_value = coin_num.loc[last_mi] * (
                                    short_fut.loc[mi, coin_name] + long_fut.loc[mi, coin_name])
                        upper_limit = total_basket_value * 0.275
                        lower_limit = total_basket_value * 0.20

                        if margin_common.loc[mi] > upper_limit.values:
                            enlarge_amount = ((margin_common.loc[mi] / 0.25) - total_basket_value)
                            enlarge_num = enlarge_amount / (
                                        short_fut.loc[mi, coin_name] + long_fut.loc[mi, coin_name])
                            coin_num.loc[mi] = (coin_num.loc[last_mi].values + enlarge_num.values)

                        elif margin_common.loc[mi] < lower_limit.values:
                            shrink_amount = (total_basket_value - (margin_common.loc[mi] / 0.25))
                            shrink_num = shrink_amount / (short_fut.loc[mi, coin_name] + long_fut.loc[mi, coin_name])
                            coin_num.loc[mi] = np.float64(coin_num.loc[last_mi] - shrink_num)


                        else:
                            coin_num.loc[mi] = coin_num.loc[last_mi]

                            binance_holdprice.loc[mi] = short_fut.loc[last_mi, coin_name]

                        okex_transaction_fee = np.sum(
                            abs((coin_num.loc[mi]).sub(coin_num.loc[last_mi], fill_value=0)) * long_fut.loc[
                                mi, coin_name] * 0.0007)
                        binance_transaction_fee = np.sum(
                            abs((coin_num.loc[mi]).sub(coin_num.loc[last_mi], fill_value=0)) * long_fut.loc[
                                mi, coin_name] * 0.0007)
                        trade_cost.loc[mi] = (binance_transaction_fee + okex_transaction_fee)
                        margin_common.loc[mi] = margin_common.loc[mi] - binance_transaction_fee - okex_transaction_fee
                        if margin_common.loc[mi] < 0 or margin_common.loc[mi] / margin_common.loc[last_mi] > 1.1 or \
                                margin_common.loc[mi] / margin_common.loc[last_mi] < 0.9 or np.isnan(margin_common.loc[mi]):
                            margin_common.loc[mi] = margin_common.loc[last_mi]
                            coin_num.loc[mi] = coin_num.loc[last_mi]
                            realized_pnl = pd.Series(index=standard_index, dtype=float)
                            coin_num.loc[mi] = coin_num.loc[last_mi]

                    # print(coin_name)
                    margin_l.append(margin_common)
                    coinnum_l.append(coin_num)
                okex_margin = pd.concat(margin_l, axis=1)
                okex_coinnum = pd.concat(coinnum_l, axis=1)
                okex_margin.columns = col_names
                okex_coinnum.columns = col_names
                okex_total = okex_margin + okex_coinnum * (long_fut - short_fut)
                okex_margin.to_pickle(margin_path)
                okex_coinnum.to_pickle(coinnum_path)
                (okex_total / okex_total.shift(1)).cumprod().to_pickle(total_path)
                (short_fut / long_fut).to_pickle(spread_path)
                print('update successfully')

    def exe_non_ftx_sub(self):

        results = []
        pool = Pool(30)
        for short in self.exchanges:
            for long in self.exchanges:
                if short=='ftx' or long=='ftx' or (long==short) :
                    continue
                else:
                    results.append(pool.apply_async(self.non_ftx_sub, args=(short, long)))
        pool.close()
        pool.join()

    def ftx_long_sub(self, short):
        long='ftx'
        if env == 'TDS':
            out_folder = '/home/strategy/usdt_spread/'
        else:
            out_folder = '/home/pro_strategy/usdt_spread/'

        coinnum_path = out_folder + 'coinnum/' + '{}2{}'.format(short,long) + '_coinnum.pkl'
        margin_path = out_folder + 'margin/' + '{}2{}'.format(short,long) + '_margin.pkl'
        spread_path = out_folder + 'spread/' + '{}2{}'.format(short,long) + '_spread.pkl'
        total_path = out_folder + 'total/' + '{}2{}'.format(short,long) + '_total.pkl'
        print(short+long)

        short_fut = self.price_data[short].T.dropna(how='all').T.loc['2021-06-01':]
        short_rate = self.funding_data[short].T.dropna(how='all').T.loc['2021-06-01':]

        long_fut = self.price_data[long].T.dropna(how='all').T.loc['2021-06-01':]
        long_rate = self.funding_data[long].T.dropna(how='all').T.loc['2021-06-01':]

        if self.env == 'MIX':
            live_rate_path = '/home/raw_data/funding/live_data/'
            live_price_path = '/home/raw_data/futures/live_data/'
            live_rate_file_path = os.path.join(live_rate_path, np.sort(os.listdir(live_rate_path))[-1])
            live_price_file_path = os.path.join(live_price_path, np.sort(os.listdir(live_price_path))[-1])
            short_new_rate = create_new_rate(live_rate_file_path, 'binance')
            short_new_fut = create_new_price(live_price_file_path, 'binance')
            for col in short_new_fut.columns:
                short_new_fut.rename(columns={col: col.split('/')[0]}, inplace=True)
                short_new_rate.rename(columns={col: col.split('/')[0]}, inplace=True)
            short_fut = short_fut.loc[:short_new_fut.index[0]]
            short_rate = short_rate.loc[:short_new_rate.index[0]]
            short_fut = pd.concat([short_fut, short_new_fut], sort=True)
            short_rate = pd.concat([short_rate, short_new_rate], sort=True)

            long_new_rate = create_new_rate(live_rate_file_path, long)
            long_new_fut = create_new_price(live_price_file_path, long)
            long_rate = long_rate.loc[:long_new_rate.index[0]]
            long_fut = long_fut.loc[:long_new_fut.index[0]]
            long_rate = pd.concat([long_rate, long_new_rate], sort=True)
            long_fut = pd.concat([long_fut, long_new_fut], sort=True)

        end_date = min(short_fut.index[-1], short_rate.index[-1], long_fut.index[-1], long_rate.index[-1])
        short_fut = short_fut.loc[:end_date]
        short_rate = short_rate.loc[:end_date]
        long_fut = long_fut.loc[:end_date]
        long_rate = long_rate.loc[:end_date]

        col_names = np.sort(list(set(long_fut.columns) & set(short_rate.columns)))
        short_fut = short_fut[col_names]
        short_rate = short_rate[col_names]
        long_fut = long_fut[col_names]
        long_rate = long_rate[col_names]
        if len(short_rate) != len(long_fut):
            print('DataProblem Warning: this will excute with intersection index')
            slt_idx = list(set(long_rate.index) & set(long_fut.index) & set(short_fut.index) & set(short_rate.index))
            long_rate = long_rate.loc[slt_idx]
            long_fut = long_fut.loc[slt_idx]
            short_fut = short_fut.loc[slt_idx]
            short_rate = short_rate.loc[slt_idx]

        if not (os.path.exists(coinnum_path) and os.path.exists(margin_path) and os.path.exists(
                spread_path) and os.path.exists(total_path)):
            print('creating new table')
            margin_l = []
            coinnum_l = []
            i = 0
            for coin_name in col_names:
                try:
                    start_date = max(long_fut[coin_name].dropna().index[0], short_fut[coin_name].dropna().index[0],
                                     long_rate[coin_name].dropna().index[0], short_rate[coin_name].dropna().index[0])
                    standard_index = long_fut[coin_name].loc[start_date:].replace(0, np.nan).dropna().index.to_list()
                    margin_common = pd.Series(index=standard_index, dtype=float)
                    margin_common.iloc[0] = 0.5
                    realized_pnl = pd.Series(index=standard_index, dtype=float)
                    funding = pd.Series(index=standard_index, dtype=float)
                    trade_cost = pd.Series(index=standard_index, dtype=float)
                    coin_num = pd.DataFrame(index=standard_index, columns=[coin_name])
                    coin_num.iloc[0] = 1 / ((short_fut[coin_name]).replace(0, np.nan).dropna().iloc[0])
                    binance_holdprice = pd.DataFrame(index=standard_index, columns=[coin_name])
                    ftx_holdprice = pd.DataFrame(index=standard_index, columns=[coin_name])
                    binance_holdprice.iloc[0] = short_fut.loc[standard_index[1], coin_name]
                    ftx_holdprice.iloc[0] = long_fut.loc[standard_index[1], coin_name]
                    idx = 0
                except:
                    margin_l.append(margin_common)
                    coinnum_l.append(coin_num)
                    continue
                for mi in standard_index[1:]:

                    idx += 1
                    last_mi = standard_index[idx - 1]

                    if mi.hour % 8 != 0:
                        funding_from_ftx = ((-long_rate.loc[mi, coin_name]) * (coin_num.loc[last_mi]) * (
                        long_fut.loc[mi, coin_name])).sum()
                        margin_common.loc[mi] = margin_common.loc[last_mi] + funding_from_ftx
                        funding.loc[mi] = funding_from_ftx
                    else:
                        funding_from_ftx = ((-long_rate.loc[mi, coin_name]) * (coin_num.loc[last_mi]) * (
                        long_fut.loc[mi, coin_name])).sum()
                        funding_from_binance = ((short_rate.loc[mi, coin_name]) * (coin_num.loc[last_mi]) * (
                        short_fut.loc[mi, coin_name])).sum()
                        margin_common.loc[mi] = margin_common.loc[last_mi] + funding_from_ftx + funding_from_binance
                        funding.loc[mi] = funding_from_ftx + funding_from_binance
                    future_pnl = coin_num.loc[last_mi] * (
                                long_fut.loc[mi, coin_name] - (long_fut.loc[last_mi, coin_name]) + short_fut.loc[
                            last_mi, coin_name] - short_fut.loc[mi, coin_name])
                    realized_pnl.loc[mi] = future_pnl.values
                    margin_common.loc[mi] = np.float64(margin_common.loc[mi] + future_pnl)
                    total_basket_value = coin_num.loc[last_mi] * (
                                short_fut.loc[mi, coin_name] + long_fut.loc[mi, coin_name])
                    upper_limit = total_basket_value * 0.275
                    lower_limit = total_basket_value * 0.2

                    if margin_common.loc[mi] > upper_limit.values:
                        enlarge_amount = ((margin_common.loc[mi] / 0.25) - total_basket_value)
                        enlarge_num = enlarge_amount / (short_fut.loc[mi, coin_name] + long_fut.loc[mi, coin_name])
                        coin_num.loc[mi] = (coin_num.loc[last_mi].values + enlarge_num.values)

                    elif margin_common.loc[mi] < lower_limit.values:
                        shrink_amount = (total_basket_value - (margin_common.loc[mi] / 0.25))
                        shrink_num = shrink_amount / (short_fut.loc[mi, coin_name] + long_fut.loc[mi, coin_name])
                        coin_num.loc[mi] = np.float64(coin_num.loc[last_mi] - shrink_num)


                    else:
                        coin_num.loc[mi] = coin_num.loc[last_mi]
                        ftx_holdprice.loc[mi] = long_fut.loc[last_mi, coin_name]
                        binance_holdprice.loc[mi] = short_fut.loc[last_mi, coin_name]

                    ftx_transaction_fee = np.sum(
                        abs((coin_num.loc[mi]).sub(coin_num.loc[last_mi], fill_value=0)) * long_fut.loc[
                            mi, coin_name] * 0.0007)
                    binance_transaction_fee = np.sum(
                        abs((coin_num.loc[mi]).sub(coin_num.loc[last_mi], fill_value=0)) * long_fut.loc[
                            mi, coin_name] * 0.0007)
                    trade_cost.loc[mi] = (binance_transaction_fee + ftx_transaction_fee)
                    margin_common.loc[mi] = margin_common.loc[mi] - binance_transaction_fee - ftx_transaction_fee
                    if (margin_common.loc[mi] < 0 or margin_common.loc[mi] / margin_common.loc[last_mi] > 1.1 or
                            margin_common.loc[mi] / margin_common.loc[last_mi] < 0.9) or np.isnan(margin_common.loc[mi]):
                        margin_common.loc[mi] = margin_common.loc[last_mi]
                        realized_pnl = pd.Series(index=standard_index, dtype=float)
                        coin_num.loc[mi] = coin_num.loc[last_mi]

                i+=1
                if i%30 ==0:
                    print('Progress{} %'.format(round(i/len(col_names),3)*100))
                margin_l.append(margin_common)
                coinnum_l.append(coin_num)
            print('finished')
            long_margin = pd.concat(margin_l, axis=1)
            long_coinnum = pd.concat(coinnum_l, axis=1)
            long_margin.columns = col_names
            long_coinnum.columns = col_names
            long_total = long_margin + long_coinnum * (long_fut - short_fut)
            long_margin.to_pickle(margin_path)
            long_coinnum.to_pickle(coinnum_path)
            (long_total / long_total.shift(1)).cumprod().to_pickle(total_path)
            (short_fut / long_fut).to_pickle(spread_path)
        else:
            print('updating existing table')
            coinnum = pd.read_pickle(coinnum_path)
            margin = pd.read_pickle(margin_path)
            col_names = np.sort(list(set(col_names) & set(coinnum.columns)))

            long_rate = long_rate[col_names]
            long_fut = long_fut[col_names]
            short_fut = short_fut[col_names]
            short_rate = short_rate[col_names]

            coinnum.index = pd.to_datetime(coinnum.index)
            margin.index = pd.to_datetime(margin.index)
            coinnum = coinnum.reindex(short_fut.index)
            margin = margin.reindex(short_fut.index)

            margin_l = []
            coinnum_l = []
            for coin_name in col_names:
                try:
                    start_date = max(long_fut[coin_name].dropna().index[0], short_fut[coin_name].dropna().index[0],
                                     long_rate[coin_name].dropna().index[0], short_rate[coin_name].dropna().index[0])
                    standard_index = long_fut[coin_name].loc[start_date:].replace(0, np.nan).dropna().index.to_list()


                    start_idx = standard_index.index(coinnum.dropna(how='all').index[-1])
                    margin_common = margin[coin_name]
                    realized_pnl = pd.Series(index=standard_index, dtype=float)
                    funding = pd.Series(index=standard_index, dtype=float)
                    trade_cost = pd.Series(index=standard_index, dtype=float)
                    coin_num = pd.DataFrame(coinnum[coin_name])
                    binance_holdprice = pd.DataFrame(index=standard_index, columns=[coin_name])
                    ftx_holdprice = pd.DataFrame(index=standard_index, columns=[coin_name])
                    binance_holdprice.iloc[0] = short_fut.loc[standard_index[1], coin_name]
                    ftx_holdprice.iloc[0] = long_fut.loc[standard_index[1], coin_name]
                    idx = start_idx - self.days * 24
                except:
                    if standard_index[-1]<coinnum.dropna(how='all').index[-1]:
                        margin_common = margin[coin_name].loc[:standard_index[-1]]
                        coin_num = pd.DataFrame(coinnum[coin_name]).loc[:standard_index[-1]]
                        margin_l.append(margin_common)
                        coinnum_l.append(coin_num)
                    else:
                        margin_common = margin[coin_name]
                        coin_num = pd.DataFrame(coinnum[coin_name])
                        margin_l.append(margin_common)
                        coinnum_l.append(coin_num)
                    continue



                for mi in standard_index[start_idx + 1 - self.days * 24:]:

                    idx += 1
                    last_mi = standard_index[idx - 1]

                    if mi.hour % 8 != 0:
                        funding_from_ftx = ((-long_rate.loc[mi, coin_name]) * (coin_num.loc[last_mi]) * (
                        long_fut.loc[mi, coin_name])).sum()
                        margin_common.loc[mi] = margin_common.loc[last_mi] + funding_from_ftx
                        funding.loc[mi] = funding_from_ftx
                    else:
                        funding_from_ftx = ((-long_rate.loc[mi, coin_name]) * (coin_num.loc[last_mi]) * (
                        long_fut.loc[mi, coin_name])).sum()
                        funding_from_binance = ((short_rate.loc[mi, coin_name]) * (coin_num.loc[last_mi]) * (
                        short_fut.loc[mi, coin_name])).sum()
                        margin_common.loc[mi] = margin_common.loc[last_mi] + funding_from_ftx + funding_from_binance
                        funding.loc[mi] = funding_from_ftx + funding_from_binance
                    future_pnl = coin_num.loc[last_mi] * (
                                long_fut.loc[mi, coin_name] - (long_fut.loc[last_mi, coin_name]) + short_fut.loc[
                            last_mi, coin_name] - short_fut.loc[mi, coin_name])
                    realized_pnl.loc[mi] = future_pnl.values
                    margin_common.loc[mi] = np.float64(margin_common.loc[mi] + future_pnl)
                    total_basket_value = coin_num.loc[last_mi] * (
                                short_fut.loc[mi, coin_name] + long_fut.loc[mi, coin_name])
                    upper_limit = total_basket_value * 0.275
                    lower_limit = total_basket_value * 0.2

                    if margin_common.loc[mi] > upper_limit.values:
                        enlarge_amount = ((margin_common.loc[mi] / 0.25) - total_basket_value)
                        enlarge_num = enlarge_amount / (short_fut.loc[mi, coin_name] + long_fut.loc[mi, coin_name])
                        coin_num.loc[mi] = (coin_num.loc[last_mi].values + enlarge_num.values)

                    elif margin_common.loc[mi] < lower_limit.values:
                        shrink_amount = (total_basket_value - (margin_common.loc[mi] / 0.25))
                        shrink_num = shrink_amount / (short_fut.loc[mi, coin_name] + long_fut.loc[mi, coin_name])
                        coin_num.loc[mi] = np.float64(coin_num.loc[last_mi] - shrink_num)


                    else:
                        coin_num.loc[mi] = coin_num.loc[last_mi]
                        ftx_holdprice.loc[mi] = long_fut.loc[last_mi, coin_name]
                        binance_holdprice.loc[mi] = short_fut.loc[last_mi, coin_name]

                    ftx_transaction_fee = np.sum(
                        abs((coin_num.loc[mi]).sub(coin_num.loc[last_mi], fill_value=0)) * long_fut.loc[
                            mi, coin_name] * 0.0007)
                    binance_transaction_fee = np.sum(
                        abs((coin_num.loc[mi]).sub(coin_num.loc[last_mi], fill_value=0)) * long_fut.loc[
                            mi, coin_name] * 0.0007)
                    trade_cost.loc[mi] = (binance_transaction_fee + ftx_transaction_fee)
                    margin_common.loc[mi] = margin_common.loc[mi] - binance_transaction_fee - ftx_transaction_fee
                    if margin_common.loc[mi] < 0 or margin_common.loc[mi] / margin_common.loc[last_mi] > 1.1 or \
                            margin_common.loc[mi] / margin_common.loc[last_mi] < 0.9 or np.isnan(margin_common.loc[mi]):
                        margin_common.loc[mi] = margin_common.loc[last_mi]
                        realized_pnl = pd.Series(index=standard_index, dtype=float)
                        coin_num.loc[mi] = coin_num.loc[last_mi]


                margin_l.append(margin_common)
                coinnum_l.append(coin_num)
            long_margin = pd.concat(margin_l, axis=1)
            long_coinnum = pd.concat(coinnum_l, axis=1)
            long_margin.columns = col_names
            long_coinnum.columns = col_names
            long_total = long_margin + long_coinnum * (long_fut - short_fut)
            long_margin.to_pickle(margin_path)
            long_coinnum.to_pickle(coinnum_path)
            (long_total / long_total.shift(1)).cumprod().to_pickle(total_path)
            (short_fut / long_fut).to_pickle(spread_path)
            print('update successfully')

    def ftx_long_sub_exe(self):
        pool = Pool(6)
        results = []
        print('ftx start')
        for exe in self.exchanges:

            if exe=='ftx':
                continue
            else:
                results.append(pool.apply_async(self.ftx_long_sub, args=(exe,)))
        pool.close()
        pool.join()
        print('ftx_end')


def main():
    generator = Sub_generator(env, days)
    generator.exe_non_ftx_sub()
    generator.ftx_long_sub_exe()


if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option('-v', dest='ENV', default="TDS")
    parser.add_option('-d', dest='Days_back', type='int', default=None)
    (options, args) = parser.parse_args()

    # 创建io对象
    days = options.Days_back
    env = options.ENV

    if env == 'TDS':
        if days is None:
            days = 5
        # 创建Path对象

    elif env == 'MIX':
        if days is None:
            days = 2
        # 创建Path对象


    script_nm = f'{substra_name}_{env}'
    run_start = dt.datetime.now()
    main()
    run_end = dt.datetime.now()
    total_run_time = (run_end - run_start).total_seconds()
    print('\n----------------------\n')
    print(script_nm, " Run time:", (total_run_time / 60).__round__(2), "mins")






