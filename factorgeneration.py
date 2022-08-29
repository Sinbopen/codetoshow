import sys
sys.path.append('/mnt/mfs/open_lib/')
import shared_tools.back_test as bt
from shared_tools.io import AZ_IO
from shared_tools.back_test import AZ_Row_zscore
from shared_utils.config.config_global import Env
from shared_tools.send_email import *
import datetime as dt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import multiprocessing
import time
from scipy.stats.mstats import winsorize
from optparse import OptionParser
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

process_num = 20
file_type = 'pkl'
io_obj = AZ_IO(mod=file_type)
outpath = '/media/hdd1/barra_factor/'

a = 0.5**(1/126)
W = np.flipud(np.array([a**x for x in np.arange(1,253,1)]))
SIM_NAME = 'BARRAS'

class Path:
    def __init__(self, mod):
        Env_obj = Env(mod=mod)
        self.tradedates_path = Env_obj.BinFiles.EM_Funda / "DERIVED_CONSTANTS/TradeDates.{}".format(file_type)
        self.aadj_r_path = Env_obj.BinFiles.EM_Funda / "DERIVED_14/aadj_r.{}".format(file_type)
        self.index_r_path = Env_obj.BinFiles.EM_Funda / "DERIVED_141" / "ZZ500_r.{}".format(file_type)
        self.market_cap_path = Env_obj.BinFiles.EM_Funda / "LICO_YS_STOCKVALUE/AmarketCap.{}".format(file_type)
        self.book_equity_path = Env_obj.BinFiles.EM_Funda / "daily/R_SUMSHEQUITY_First.{}".format(file_type)
        self.turn_rate_path = Env_obj.BinFiles.EM_Funda / "TRAD_SK_DAILY_JC/TURNRATE.{}".format(file_type)
        self.book_asset_path = Env_obj.BinFiles.EM_Funda / "daily/R_SUMASSET_First.{}".format(file_type)
        self.book_lia_path = Env_obj.BinFiles.EM_Funda / "daily/R_SUMLIAB_First.{}".format(file_type)
        self.net_cashflow_path = Env_obj.BinFiles.EM_Funda / "daily/R_NetCf_TTM_First.{}".format(file_type)
        self.PE_TTM_path = Env_obj.BinFiles.EM_Funda/ "TRAD_SK_REVALUATION/PE_TTM.{}".format(file_type)
        self.Longtemp_Diab_path = '/mnt/mfs/DAT_EQT/EM_Funda/daily/R_NONLLIABONEYEAR_First.{}'.format(file_type)
        self.pe_fy1_path = Env_obj.BinFiles.EM_Funda/ "DERIVED_CHQ/FFY/PE_FY_mean_1.dat"
        self.eps_path = Env_obj.BinFiles.EM_Funda/ "daily/R_EPSBasicPS_First.{}".format(file_type)
        self.operaterev_PS_path = Env_obj.BinFiles.EM_Funda/ "daily/R_RevenuePS_s_First.{}".format(file_type)
        self.ana_profit_path = Env_obj.BinFiles.EM_Funda/ "DERIVED_CHQ/FFY/NetProfit_FY_mean_1.dat"
        self.netprofit_path = Env_obj.BinFiles.EM_Funda/"daily/R_NETPROFIT_First.{}".format(file_type)
        self.StPtStock_path = Env_obj.BinFiles.EM_Funda / "DERIVED_01/StAndPtStock.pkl"
#实例化Path需要输入模式{bkt， Pro，bkt_test,pro_test}

#io_obj

class Barra:

    def __init__(self):
        self.filter =  io_obj.load_data(Path_obj.StPtStock_path).loc["2014-01-04":]
        self.aadj_r = io_obj.load_data(Path_obj.aadj_r_path).loc["2014-01-04":]
        self.index_r = io_obj.load_data(Path_obj.index_r_path).loc["2014-01-04":]
        self.mod_indexr = self.index_r.reindex(self.aadj_r.index)
        self.market_cap = io_obj.load_data(Path_obj.market_cap_path).loc["2014-01-04":]
        self.tradedate = io_obj.load_data(Path_obj.tradedates_path).loc["2014-01-04":]

    def size(self):
        log_market = np.log(self.market_cap)* self.filter
        # log_market.to_pickle("/mnt/mfs/dat_zym/SIZE.pkl")
        log_market = AZ_Row_zscore(log_market, cap=3.5).loc['2015-01-05':]
        io_obj.save_data(log_market, outpath+"size_factor.pkl")
        print("size Finished and saved")
        return log_market

    def momentum(self):
        a =  0.5 ** (1 / 126)
        W = np.flipud(np.array([a ** x for x in np.arange(21, 525, 1)]))
        aadj = io_obj.load_data(Path_obj.aadj_r_path).loc["2012-01-04":]
        ewma = np.log(aadj +1).rolling(window=525).apply(lambda x: np.sum(x[:504]*W))*self.filter
        ewma = AZ_Row_zscore(ewma, cap=3.5).loc['2015-01-05':]
        io_obj.save_data(ewma, outpath+"momentum_factor.pkl")
        print("momentum Finished and saved")
        return ewma

    def _wls_reg(self, row):

        model = sm.WLS(row.iloc[:,0], sm.add_constant(row.iloc[:,1]), weights=W).fit()
        if len(model.params)== 0 or len(model.params)==1:
            return np.nan
        else:
            return model.params[1]

    def _rolling_reg(self, stkret, indexret, stk):

        res = pd.DataFrame(index = self.aadj_r.index, columns=[stk])
        for i in np.arange(252,len(stkret.index),1):
            res[stk].loc[stkret.index[i]]= sm.WLS(stkret[i-252:i], indexret[i-252:i], weight = W).fit().params[0]
        return res

    def beta(self):
        beta_outpath = os.path.join(outpath, 'beta_factor.pkl')

        if os.path.exists(beta_outpath) is False:
            betas = pd.DataFrame(columns=self.aadj_r.columns, index=self.aadj_r.index)
            stks = list(self.aadj_r.columns)
            results = []
            res_list = []
            pool = Pool(10)
            for stk in stks:
                results.append(pool.apply_async(self._rolling_reg, args=(self.aadj_r[stk], self.index_r, stk)))
            for res in results:
                res_list.append(res.get())
            betas = pd.concat(res_list, axis=1, sort=False)
            betas = betas.T.sort_index(axis=0).T
            betas = betas *self.filter
            betas = AZ_Row_zscore(betas, cap=3.5)

        else:

            stk_r_last_date = io_obj.load_data(Path_obj.aadj_r_path).index[-days]
            beta_last_date = io_obj.load_data(beta_outpath).index[-1]
            start_date = min(stk_r_last_date, beta_last_date)
            sel_startdate = self.aadj_r.index[self.aadj_r.index.get_loc(start_date, method='bfill')-252]
            sel_aadj_r = self.aadj_r.loc[sel_startdate:]
            sel_index_r = self.index_r.loc[sel_startdate:]
            stks = list(sel_aadj_r.columns)
            results = []
            for stk in stks:
                results.append(self._rolling_reg(sel_aadj_r[stk],sel_index_r, stk))
            new_betas = pd.concat(results, axis=1, sort=False)
            new_betas = new_betas.T.sort_index(axis=0).T
            new_betas = new_betas * self.filter
            new_betas = AZ_Row_zscore(new_betas, cap=3.5)
            old_betas = io_obj.load_data(beta_outpath)
            new_betas = new_betas[new_betas.index >= start_date]
            betas = bt.AZ_Factor_Combine(old_betas, new_betas)

        io_obj.save_data(betas, beta_outpath)
        print("beta Finished and saved")
        return betas



    # model = LinearRegression()
    # model.fit(index_r, aadj_r)

    def res_vol(self):
#0.74*datsd + 0.16*cmra +0.1*hsigma
        h = 42
        T = 252
        sigma = 0.5**(1/h)
        dastd = pd.DataFrame.reindex(self.aadj_r)
        weight = np.array([sigma ** (T-i-1) for i in range(T)])
        for day in self.aadj_r.index[252:]:
            tradedate_sel = self.tradedate.index[self.tradedate.index.get_loc(day)-T : self.tradedate.index.get_loc(day)]
            temp = self.aadj_r.loc[tradedate_sel]
            temp = (temp - temp.mean())**2
            dastd.loc[day] = 1/T * np.matmul(weight.T, temp)


#For the past 12 months, substitute with 252 trading days
        ZT = []
        for i in np.arange(1,13):
            ZT.append( np.log(1+self.aadj_r).rolling(window=i*21).sum().values)
        ZT = np.array(ZT)
        cmra_df =pd.DataFrame(np.log(1+ZT.max(axis=0))-np.log(1+ZT.min(axis=0)))
        cmra_df.columns = self.aadj_r.columns
        cmra_df.index = self.aadj_r.index
# 生成beta回归残差
        res_beta = self.aadj_r
        new_df = pd.concat([self.aadj_r,self.index_r], axis = 1)
        x = new_df.iloc[:,-1]
        for col in self.aadj_r.columns:
            beta = np.sum(x*self.aadj_r[col])/np.sum(x**2)
            alpha = self.aadj_r[col].mean() - beta*x.mean()
            residuals = self.aadj_r[col] - beta*x - alpha
            res_beta[col] = np.std(residuals)

        dastd = AZ_Row_zscore(dastd*self.filter, cap=3.5)
        cmra_df = AZ_Row_zscore(cmra_df*self.filter, cap=3.5)
        res_beta = AZ_Row_zscore(res_beta*self.filter, cap=3.5)

        resvol = AZ_Row_zscore((0.74*dastd+ 0.16*cmra_df+ 0.1*res_beta), cap=3.5).loc['2015-01-05':]
        io_obj.save_data(resvol , outpath + "resvol_factor.pkl")
        print("res_vol Finished and saved")
        return resvol




    def nlsize(self):
        size = np.log(self.market_cap)*self.filter
        cube_size = size**3
        nlsize_df = size
        for col in size.columns:
            beta = np.sum(cube_size[col] * size[col]) - np.sum(size[col])*np.mean(cube_size[col]) / (np.sum(size[col] ** 2)+np.sum(size[col])*np.mean(size[col]))
            alpha = cube_size[col].mean() - beta * size[col].mean()
            residuals = cube_size[col] - beta *size[col] - alpha
#将上下5%的数去极值
            s_residuals = (residuals - np.mean(residuals))/np.std(residuals)
            nlsize_df[col] = winsorize(s_residuals, limits=[0.05,0.05])
        nlsize_df = nlsize_df * self.filter
        nlsize_df = AZ_Row_zscore(nlsize_df, cap=3.5).loc['2015-01-05':]
        io_obj.save_data(nlsize_df, outpath + "nlsize_factor.pkl")
        print("nlsize Finished and saved")
        return nlsize_df


    def btop(self):
#book value /market value
        bookvalue_df = io_obj.load_data(Path_obj.book_asset_path)
        btop = (bookvalue_df / self.market_cap)
        btop = btop*self.filter
        btop = AZ_Row_zscore(btop, cap=3.5).loc['2015-01-05':]
        io_obj.save_data(btop, outpath + "btop_factor.pkl")
        print("btop Finished and saved")
        return btop



    def liquid(self):
#0.35*stom + 0.35*stog +0.3*stoa
#share turnover one month
#
        turn_rate_df = io_obj.load_data(Path_obj.turn_rate_path)
        stom_df = np.log(turn_rate_df.rolling(window =21).sum())
        stoq_df = np.exp(stom_df).rolling(window = 3).mean()
        stoa_df = np.log(np.exp(stom_df).rolling(window=12).mean())

        stom_df = stom_df.sub(stom_df.mean(1), axis=0).div(stom_df.std(1).replace(0,np.nan), axis=0)
        stoq_df = stoq_df.sub(stoq_df.mean(1), axis=0).div(stoq_df.std(1).replace(0,np.nan), axis=0)
        stoa_df = stoa_df.sub(stoa_df.mean(1), axis=0).div(stoa_df.std(1).replace(0,np.nan), axis=0)

        stom_df[stom_df>3.5] = 3.5
        stom_df[stom_df<-3.5] = -3.5
        stoq_df[stoq_df>3.5] = 3.5
        stoq_df[stoq_df<-3.5] = -3.5
        stoa_df[stoa_df>3.5] = 3.5
        stoa_df[stoa_df<-3.5] = -3.5

        liquid = 0.35*stom_df.add(0.35*stoq_df, fill_value=0).add(0.3*stoa_df, fill_value=0)
        liquid = liquid * self.filter
        liquid = AZ_Row_zscore(liquid, cap=3.5).loc['2015-01-05':]
        io_obj.save_data(liquid, outpath + "liquid_factor.pkl")
        print("liquid Finished and saved")
        return liquid



    def leverage(self):
#mlev是市场杠杆，(me+pe+ld)/me， 普通股市值，优先股账面价值，长期负债账面价值
#A股市场并没有优先股，所以这里只研究A股市场，优先股的市值和账面价值都为0，长期负债由非流动负债代替
#pe = 0, me=市值
        me_df = self.market_cap
        ld_df =io_obj.load_data(Path_obj.Longtemp_Diab_path)
        mlev = (me_df+ld_df)/me_df


#账面资产负债比dtoa
        ta = io_obj.load_data(Path_obj.book_asset_path)
        td = io_obj.load_data(Path_obj.book_lia_path)
        dtoa = ta/td
#账面杠杆
# (be+pe+ld)/be = (be+ld)/be
        be_df = io_obj.load_data(Path_obj.book_equity_path)
        blev = (be_df + ld_df)/be_df

        mlev= AZ_Row_zscore(mlev* self.filter, cap=3.5)
        dtoa= AZ_Row_zscore(dtoa* self.filter, cap=3.5)
        blev= AZ_Row_zscore(blev* self.filter, cap=3.5)

        leverage = AZ_Row_zscore(0.38*mlev +0.35*dtoa+0.27*blev).loc['2015-01-05':]
        io_obj.save_data(leverage, outpath + "leverage_factor.pkl")
        print("leverage Finished and saved")
        return leverage

    def earning_yeild(self):
#epfwd  预期盈利市值比(分析师),分析师数据密度比较低。
        epfwd = 1/ io_obj.load_data(Path_obj.pe_fy1_path)
#cetop  现金流量市值比
        net_cashflow_df = io_obj.load_data(Path_obj.net_cashflow_path)
        cetop = net_cashflow_df/self.market_cap
#etop   盈利市值比
        etop =1/ io_obj.load_data(Path_obj.PE_TTM_path)

        epfwd = AZ_Row_zscore(epfwd*self.filter, cap=3.5)
        cetop = AZ_Row_zscore(cetop*self.filter, cap=3.5)
        etop = AZ_Row_zscore(etop*self.filter, cap=3.5)

        earning_yield = 0.68*epfwd.add(0.21*cetop, fill_value=0).add(0.11*etop, fill_value=0)
        earning_yield = AZ_Row_zscore(earning_yield *self.filter, cap=3.5).loc['2015-01-05':]
        io_obj.save_data(earning_yield, outpath + "earnyield_factor.pkl")
        print("earning_yield Finished and saved")
        return earning_yield

    def _reg(self,y):
        x = np.arange(1,1826,1)

        beta = np.sum((x - x.mean())*(y - y.mean()))/np.sum((x-x.mean())**2)
        return beta



    def growth(self):
#egrlf  未来3-5年分析师预期盈利增长率
        egrlf = 0
#egrsf  未来1年分析师预期盈利增长率，使用预期盈利来计算预期增长率

        ana_profit = io_obj.load_data(Path_obj.ana_profit_path)
        net_profit = io_obj.load_data(Path_obj.netprofit_path)
        egrsf = ana_profit/net_profit
#egro   过去5年盈利增长率
#过去五年粗略为1825自然日，表格R_EPSBasicPS_First.pkl的index为自然日，计算方式为每股收益对时间回归的斜率/平均每股年收益

        eps_df = io_obj.load_data(Path_obj.eps_path)
        egro = eps_df.rolling(window=1825).apply(self._reg)

#sgro   过去5年营业收入增长,与盈利增长相同的方式
        operate_rev_PS = io_obj.load_data(Path_obj.operaterev_PS_path)
        sgro = operate_rev_PS.rolling(window=1825).apply(self._reg)/operate_rev_PS

        egrsf = AZ_Row_zscore(egrsf * self.filter, cap=3.5)
        egro = AZ_Row_zscore(egro* self.filter, cap=3.5)
        sgro = AZ_Row_zscore(sgro* self.filter, cap=3.5)
        growth = AZ_Row_zscore(0.11*egrsf.add(0.24*egro, fill_value=0).add(0.47*sgro, fill_value=0), cap=3.5).loc['2015-01-05':]
        io_obj.save_data(growth, outpath + 'growth_factor.pkl')
        print('growth_factor finished and saved')
        return growth

def main():
    barra_obj = Barra()
    barra_obj.leverage()
    barra_obj.liquid()
    barra_obj.momentum()
    barra_obj.size()
    barra_obj.nlsize()
    barra_obj.res_vol()
    barra_obj.btop()
    # barra_obj.earning_yeild()
    # barra_obj.growth()
    barra_obj.beta()
if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option('-v', dest='ENV', default="PRO")
    parser.add_option('-d', dest='Days_back', type='int', default=None)
    (options, args) = parser.parse_args()

    # 创建io对象
    io_obj = AZ_IO(mod='pkl')
    days = options.Days_back
    env = options.ENV

    if env == 'BKT':
        if days is None:
            days = 10
        # 创建Path对象
        Path_obj = Path(mod='bkt')
    elif env == 'PRO':
        if days is None:
            days = 3
        # 创建Path对象
        Path_obj = Path(mod='pro')

    script_nm = f'{SIM_NAME}_{env}'
    run_start = dt.datetime.now()
    main()
    run_end = dt.datetime.now()
    total_run_time = (run_end - run_start).total_seconds()
    print('\n----------------------\n')
    print(script_nm, " Run time:", (total_run_time / 60).__round__(2), "mins")


