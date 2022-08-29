import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/mnt/mfs/open_lib')
import datetime as dt
from datetime import  datetime
import statsmodels.api as sm
import shared_tools.back_test as bt
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shared_tools.send_email as email
from shared_tools.send_email import send_email
sys.path.append('/mnt/mfs/open_lib')
from shared_tools.io import AZ_IO
from shared_tools.Factor_Evaluation_Common_Func import AZ_Corr_test

file_type = 'pkl'
io_obj = AZ_IO(mod = file_type)
hs300 = io_obj.load_data('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_141/HS300_a.pkl')
zz500 = io_obj.load_data('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_141/ZZ500_a.pkl')
t800 = io_obj.load_data('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_141/T800_a.pkl')
zz1000 = io_obj.load_data('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_141/ZZ1000_a.pkl')
t1800 = io_obj.load_data('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_141/T1800_a.pkl')
aadj_twap_r_1015 = io_obj.load_data('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_14/aadj_twap_r_1015.pkl')
LimitedBSstk = io_obj.load_data('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_01/LimitedBuySellStock.pkl')
StStock = io_obj.load_data('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_01/StAndPtStock.pkl')
Suspendstk = io_obj.load_data('/mnt/mfs/DAT_EQT/EM_Funda/DERIVED_01/SuspendedStock.pkl')
forbid_df = LimitedBSstk* StStock* Suspendstk
recieve_ls = ['fchdao@tongyicapital.com']



def factor_expo_reg(factor, factor_l):

#factor_l 只能输入和barra因子相同的名字，且是一个装str的list

    momentum = pd.read_pickle('/mnt/mfs/temp/barra_factor/momentum_factor.pkl')
    size = pd.read_pickle('/mnt/mfs/temp/barra_factor/size_factor.pkl')
    nlsize = pd.read_pickle('/mnt/mfs/temp/barra_factor/nlsize_factor.pkl')
    resvol = pd.read_pickle('/mnt/mfs/temp/barra_factor/resvol_factor.pkl')
    liquid = pd.read_pickle('/mnt/mfs/temp/barra_factor/liquid_factor.pkl')
    btop = pd.read_pickle('/mnt/mfs/temp/barra_factor/btop_factor.pkl')
    leverage = pd.read_pickle('/mnt/mfs/temp/barra_factor/leverage_factor.pkl')
    factor = factor.dropna(axis ='index' , how='all')
    factor = factor.sub(factor.mean(1), axis=0).div(factor.std(1).replace(0, np.nan), axis =0)
    factor[factor > 3.5] = 3.5
    factor[factor < -3.5] = -3.5
    startdate = max(factor.index[0],momentum.index[0],nlsize.index[0],resvol.index[0],
                    liquid.index[0],btop.index[0], leverage.index[0], size.index[0])
    enddate = min(factor.index[-1],momentum.index[-1],nlsize.index[-1],resvol.index[-1],
                    liquid.index[-1],btop.index[-1], leverage.index[-1], size.index[-1])

    factor = factor.loc[startdate: enddate]
    momentum = momentum.loc[startdate: enddate]
    size = size.loc[startdate: enddate]
    nlsize = nlsize.loc[startdate: enddate]
    resvol = resvol.loc[startdate: enddate]
    liquid = liquid.loc[startdate: enddate]
    btop = btop.loc[startdate: enddate]
    leverage = leverage.loc[startdate: enddate]
    res = []
    dayindex = []
    for day in factor.index:

        dataset = pd.DataFrame()
        dataset = pd.concat([dataset, factor.loc[day]],axis=1, sort=True)

        for expo in factor_l:
            new_serie = eval(expo).loc[day]
            dataset = pd.concat([dataset, new_serie], axis=1, sort= True)

        dataset = dataset.dropna(axis=0, how='any')
        if len(dataset)== 0:
            continue
        model = sm.OLS(dataset.iloc[:, 0], dataset.iloc[:, 1:]).fit()
        params = np.array(model.params, ndmin=2).T

        obs = np.array(dataset.iloc[:, 1:], ndmin=2)
        regvalue = pd.Series((np.array(np.array(dataset.iloc[:, 0], ndmin=2) - obs.dot(params).T)).reshape(-1))
        regvalue.index = dataset.index
        res.append(regvalue)
        dayindex.append(day)

    res = pd.concat(res, axis=1, sort=False).T
    res.index = dayindex
    return res



#1
def chg(data, lag=1):
    type = 1
    output = (data-data.shift(lag))/data.shift(lag)
    return output

def chg2ma5(data, lag=1):
    type = 2
    ma5 = data.rolling(5).mean()
    output = (data - data.shift(lag)) / ma5
    return output

def chg2ma10(data, lag=1):
    type = 3
    ma10 = data.rolling(10).mean()
    output = (data - data.shift(lag)) / ma10
    return output


def chg2ma20(data, lag=1):
    type = 4
    ma20 = data.rolling(20).mean()
    output = (data - data.shift(lag)) / ma20
    return output

def chgg(data, lag = 1):
    type = 5
    return data - data.shift(lag)/((data+data.shift(5))/2)

def SMA(data, lag = 5):
    type = 6
    return data.rolling(lag).mean()

def raw(data, lag=1):
    type = 0
    return data

#2

def base_hs300(factor_df):

    hs300_pos = hs300 * factor_df
    hs300_factor = hs300_pos.dropna(how='all', axis='index')
    return hs300_factor

def base_zz500(factor_df):

    zz500_pos = factor_df*zz500
    zz500_factor = zz500_pos.dropna(how='all', axis='index')
    return zz500_factor

def base_zz1000(factor_df):

    zz1000_pos = factor_df * zz1000
    zz1000_factor = zz1000_pos.dropna(how='all', axis='index')
    return zz1000_factor

def base_t1800(factor_df):

    t1800_pos = factor_df * t1800
    t1800_factor = factor_df.dropna(how='all', axis='index')
    return t1800_factor

#3

#sigma-1 position
#top-bottom 10% long-short position
def equal1_pctg_10(data):
    type =1
    data = data.replace(0, np.nan)
    data = data[data.count(1)>=20]
    degree = data.apply(lambda x: pd.qcut(x, [0,0.1,0.9,1],[1,2,10], duplicates='drop'), axis = 1 )
    long_df = degree[degree==10]
    long_df = long_df.div(long_df.sum(1).replace(0, np.nan), axis = 0)
    short_df = degree[degree==1]
    short_df = short_df.div(short_df.sum(1).replace(0, np.nan), axis = 0)
    pos_df = long_df.sub(short_df, fill_value=0)
    return pos_df

#top-bottome 20% long-short position
def equal1_pctg_20(data):
    type = 2
    data = data.replace(0, np.nan)
    data = data[data.count(1)>=20]
    degree = data.apply(lambda x: pd.qcut(x, [0,0.2,0.8,1],[1,2,10], duplicates='drop'), axis = 1 )
    long_df = degree[degree==10]
    long_df = long_df.div(long_df.sum(1).replace(0, np.nan), axis = 0)
    short_df = degree[degree==1]
    short_df = short_df.div(short_df.sum(1).replace(0, np.nan), axis = 0)
    pos_df = long_df.sub(short_df, fill_value=0)
    return pos_df

#top-bottom 30% long-short position
def equal1_pctg_30(data):
    type = 3
    data = data.replace(0, np.nan)
    data = data[data.count(1) >= 20]
    degree = data.apply(
        lambda x: pd.qcut(x, [0, 0.3,0.7,1],[1,2,10], duplicates='drop'), axis=1)
    long_df = degree[degree == 10]
    long_df = long_df.div(long_df.sum(1).replace(0, np.nan), axis=0)
    short_df = degree[degree == 1]
    short_df = short_df.div(short_df.sum(1).replace(0, np.nan), axis=0)
    pos_df = long_df.sub(short_df, fill_value=0)
    return pos_df

def tag_50(row):
    indexs =  row.replace(0, np.nan).dropna().sort_values(ascending=False)[:50].index.tolist()
    top50 = pd.Series(1, index = np.sort(indexs))
    indexs =  row.replace(0, np.nan).dropna().sort_values()[:50].index.tolist()
    bottom50 = pd.Series(-1, index = np.sort(indexs))
    output = top50.add(bottom50, fill_value=0)
    return output


def equal1_rank_50(data):
    type = 4
    data = data[data.count(1) >= 20]
    degree = data.apply(lambda x: tag_50(x), axis=1)
    long = degree[degree>0]
    short = degree[degree<0]
    long = long.div(long.sum(1).replace(0,np.nan), axis=0)
    short = short.div(short.sum(1).replace(0,np.nan), axis=0)
    pos = long.sub(short, fill_value=0)
    return pos


def rowzscore_pos(data):
    type = 5
    data = data.replace(0, np.nan)
    data = data[data.count(1) >= 20]
    return bt.AZ_Row_zscore(data)

def to_final_position(factor_score, forbid_day, method_func='standard'):
    if method_func == 'simple':
        pos_fin = factor_score.shift(1) * forbid_day
        return pos_fin
    else:
        pos_fin = factor_score.shift(1) * forbid_day
        pos_fin = pos_fin.replace(np.nan, 0)
        pos_fin = pos_fin * forbid_day
        pos_fin = pos_fin.ffill()
        return pos_fin

def annual_return(pos_df, pnl_df, alpha_type):
    temp_pnl = pnl_df.sum()
    if alpha_type == 'ls_alpha':
        temp_pos = pos_df.abs().sum().sum() / 2
    else:
        temp_pos = pos_df.abs().sum().sum()
    if temp_pos == 0:
        return .0
    else:
        return temp_pnl * 250.0 / temp_pos


def factor_stats(factor_df, univ_data, index_data_r, forbid_days, rtn_df, method_func):
    '''
    :param factor_df:   factor df
    :param univ_data:   T1800 / HS300 / ZZ500 / T1000
    :param index_data_r:  series of index_rtn
    :param forbid_days:  df of tradevalid
    :param rtn_df:    df of stock rtn
    :param method_func:   feature/factor/ls_alpha/hg_alpha
    :return:  sharpe_median / leverage ratio /  pot / pnl series , depend on method func
    '''
    if method_func == 'hg_alpha':
        try:
            hedge_w = factor_df['000905.SH']
        except:
            hedge_w = factor_df['000300.SH']
        index_rtn_w = index_data_r * hedge_w.shift(2)
    factor_sel = factor_df.reindex(univ_data.columns, axis="columns") * univ_data
    factor_sel = factor_sel
    # factor_sel = factor_sel.dropna(axis=0, how='all')
    forbid_days = forbid_days.reindex_like(factor_sel)
    return_df = rtn_df.reindex_like(factor_sel)
    if method_func == 'feature' or method_func == 'factor':
        factor_z = bt.AZ_Row_zscore(factor_sel, cap=4.5)
        pos_final = to_final_position(factor_z, forbid_days)
        pnl_final = (pos_final.shift(1) * return_df).sum(axis=1)
        return pos_final, pnl_final
    elif method_func == 'ls_alpha':
        pos_final = to_final_position(factor_sel, forbid_days)
        pnl_final = (pos_final.shift(1) * return_df).sum(axis=1)
        return pos_final, pnl_final
    elif method_func == 'hg_alpha':
        pos_final = to_final_position(factor_sel, forbid_days)
        pnl_final = (pos_final.shift(1) * return_df).sum(axis=1) + index_rtn_w
        return pos_final, pnl_final

def gauge_pos(pos):

    t_cost = 0.0004
    cutoff_date = "2018-12-31"
    pos_df, dailypnl_gross = factor_stats(pos, t1800, aadj_twap_r_1015['000905.SH'], forbid_df, aadj_twap_r_1015, "ls_alpha")
    daily_cost = (pos_df.diff().abs() * t_cost).sum(axis=1)
    dailypnl_net = dailypnl_gross - daily_cost
    dailypnl_gross = dailypnl_gross.dropna().loc['2016-01-05':]
    dailypnl_net = dailypnl_net.dropna().loc['2016-01-05':]

    dailypnl_net_p1 = dailypnl_net.loc[dailypnl_net.index <= cutoff_date]
    dailypnl_gross_p1 = dailypnl_gross.loc[dailypnl_gross.index <= cutoff_date]
    pos_p1 = pos_df.loc[pos_df.index <= cutoff_date]

    dailypnl_net_p2 = dailypnl_net.loc[dailypnl_net.index > cutoff_date]
    dailypnl_gross_p2 = dailypnl_gross.loc[dailypnl_gross.index > cutoff_date]
    pos_p2 = pos_df.loc[pos_df.index > cutoff_date]

    sp1 = bt.AZ_Sharpe_quantile(dailypnl_net_p1)
    pot1 = bt.AZ_Pot(pos_p1, dailypnl_gross_p1.fillna(0).cumsum().iloc[-1])
    annr1 = annual_return(pos_p1, dailypnl_net_p1, 'ls_alpha')

    sp2 = bt.AZ_Sharpe_quantile(dailypnl_net_p2)
    pot2 = bt.AZ_Pot(pos_p2, dailypnl_gross_p2.fillna(0).cumsum().iloc[-1])
    annr2 = annual_return(pos_p2, dailypnl_net_p2, 'ls_alpha')
    return sp1, pot1, annr1, sp2, pot2, annr2

# def satisfied_submit(result):
#     output = []
#     for

def test_factor(factor):

    rawdata = factor.copy()
    neg_or_pos = [-1, 1]
    rawprocess_methods  = ['raw','chg','chg2ma5','chg2ma10','chg2ma20', 'chgg', 'SMA']
    rawprocess_lags = [1,5,10,20,60,120]
    neutral_methods = ['momentum','size','liquid', 'btop', 'raw']
    universes = ['base_hs300','base_zz500', 'base_zz1000','base_t1800']
    pos_methods = ['equal1_pctg_10', 'equal1_pctg_20', 'equal1_pctg_30', 'equal1_rank_50', 'rowzscore_pos']
    holdingday_list = [1 , 5, 10, 20, 60, 120]
    results = []
#rawdata processing
    for direction in neg_or_pos :
        factor = direction * rawdata
        for rawprocess_method in rawprocess_methods:
            for i in rawprocess_lags:
                first= eval(f'{rawprocess_method}(factor,{i})')
                for neutral_method in neutral_methods:
                    if neutral_method == 'raw':
                        second = first.copy()
                    else:
                        second = eval(f"factor_expo_reg(first, ['{neutral_method}'])")

                    for universe in universes:
                        print(f"{universe}(second)")
                        third = eval(f"{universe}(second)")

                        for pos_method in pos_methods:
                            try :
                                print(f"{pos_method}(third)")
                                forth = eval(f"{pos_method}(third)")
                            except:
                                continue
                            for j in holdingday_list:
                                final = forth.fillna(0).rolling(j).mean()
                                sp1, pot1, annr1, sp2, pot2, annr2 = gauge_pos(final)
                                print(sp1, pot1, annr1, sp2, pot2, annr2)
                                if sp1>1.2 or sp2>1.2:
                                    result = [direction,rawprocess_method,i,neutral_method,universe, pos_method,j,sp1,pot1,annr1,sp2,pot2,annr2]
                                    results.append(result)
    result_df = pd.DataFrame(results)
    results.columns = ['direction','rawprocess_method','rawprocess_lag','neutral_method', 'universe','pos_method',
                       'holding_days','sp1','pot1','annr1','sp2','pot2','annr2' ]
    return results


