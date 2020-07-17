# -*- coding: utf-8 -*-
"""
SOFR Curve Bootstrapping using constant forward or cubic spline method(nature)
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt        
        
class sofr_curve:
    
     # inputs are all np arrays, defining the instance attributes 
    
    def __init__(self, today, sofr_rlz_dates, sofr_realized, fut_dates, fut_rates, swap_tenors, 
                 swap_rates, if_linear = False):
        self.today, self.sofr_rlz_dates, self.sofr_realized, self.fut_dates, \
        self.fut_rates, self.swap_tenors, self.swap_rates, self.if_linear\
                                     = today, sofr_rlz_dates, sofr_realized, \
                                        fut_dates, fut_rates, swap_tenors, \
                                        swap_rates, if_linear
    
    # inputs are all np arrays, initialize the class variables 
    
        n = len(fut_rates) + len(swap_tenors)
        fut_expiries, _ = self._dates_to_expiries(fut_dates)
        self.pillar_expiries = np.append(fut_expiries, swap_tenors)
        self.pillar_expiries = np.insert(self.pillar_expiries, 0, 0)
              
    #  Define the discount factor array, with 1 more node denoting time zero, 
    #  the array is initialized with value of "1"
        
        self.df = np.ones((n + 1, ))
        self._optimize()
        
   # Switch between Raw Method and Cubic Spline Method
        
        if self.if_linear:
            self.curve = interp1d(self.pillar_expiries, np.log(self.df),
                                  fill_value="extrapolate")    
        else:
            self.curve = CubicSpline(self.pillar_expiries, np.log(self.df),
                                   bc_type = 'natural')
            
        self.daily_expiries,_ = self._daily_expiry(0, 40)
                
    def plot_fwd(self):
        dt = np.diff(self.daily_expiries)
        self.daily_df = np.exp(self.curve((self.daily_expiries)))
        self.daily_fwd_rates = ( self.daily_df[:-1] -  self.daily_df[1:]) / self.daily_df[1:] / dt
        plt.plot(self.daily_expiries[:-1], self.daily_fwd_rates)
        if self.if_linear:
            plt.title('SOFR FWD - Linear Interp')
            plt.savefig('linear.png', dpi=400)
        else:
            plt.title('SOFR FWD - Natural Cubic Spline Interp')
            plt.savefig('cs.png', dpi=400)
        plt.show()
        plt.clf()
        return 
    
    # For SOFR futures, convert the Futures settlement date into the futures
    # expiries
    def _dates_to_expiries(self, dates):
        '''
        input: 
            dates:      numpy array of size (n, )
                        dates beyond self.today
        output:
            expiries:   numpy array of size (n, )
                        expiries corresponding to input dates
                        
            dt:         numpy array of size (n - 1, )
                        difference in date between expiries
        '''
        dates = pd.to_datetime(dates)
        expiries = (dates - pd.to_datetime(self.today)).\
                                    astype('timedelta64[D]') / 365
        # transform back to (n,) numpy array                            
        expiries = expiries.values.reshape((len(expiries, )))
        dt = np.diff(expiries)
        return expiries, dt
    
    
    
    def _daily_expiry(self, start, end):
        '''
        start end are tenor in years
        '''
        start, end = pd.to_timedelta([start, end], unit='y')\
                            + pd.to_datetime(self.today)
        fwd_dates = pd.bdate_range(start, end)
        expiries = (fwd_dates - pd.to_datetime(self.today)).\
                                    astype('timedelta64[D]') / 365
        expiries = expiries.values.reshape((len(expiries, )))
        dt = np.diff(expiries)
        return expiries, dt
 
    
    def _log_df_to_fwd(self, maturities, log_df):
        '''
        input: 
            df_dates:       numpy array of size (n, )
                            dates corresponding to discount factors
                            including self.today
            log_df:         numpy array of size (n, )
                            log of discount factors
        output:
            swap_rate:      numpy array of size (n, )
                            swap rates corresponding to each df_dates    
        '''
        df = np.exp(log_df)
        tau = np.diff(maturities)
        return (df[:-1] - df[1:]) / tau / df[1:]        
         
    # SOFR futures valuation: the following code loops through each expiration 
    # date for SOFR futures contracts, the first SOFR futures contract will have 
    # reference rate based on both realized and forecast daily SOFR rate, thus 
    # when i==0, the treatment is different from the rest of the contracts
  
    
    def fut_value_diff(self, i, cs_func):
        if i == 0:
            start = self.today
        else:
            start = self.fut_dates[i - 1]
        end = self.fut_dates[i]
        fwd_dates = pd.bdate_range(start, end)
        expiries, dt_fwd = self._dates_to_expiries(fwd_dates)
        
        log_df = cs_func(expiries)
        df = np.exp(log_df)
        sofr_fwd = (df[:-1] - df[1:]) /  df[1:] / dt_fwd
    # SR3 will be priced based on the daily compounded SOFR rate over the
    # reference quarter
        if i == 0:
            accru_rzd = (self.sofr_realized/360 + 1).prod()
            fut_rate = (accru_rzd * \
               (sofr_fwd * dt_fwd + 1).prod()-1) * 360 / 91
        else:
            fut_rate = \
                ((sofr_fwd * dt_fwd + 1).prod()-1) * 360 / len(expiries)

        return 100*(100 - fut_rate*100  - self.fut_rates[i]) ** 2

    # SOFR Swap valuation: the following code loops through each tenor in swap 
    # segment, as the floating leg follows OIS swap convention, the daily 
    # SOFR forward rates will be compounded within the year(tenor-1, tenor) to 
    # get the final payment for the floating leg

    def swap_value_diff(self, i, cs_func):
        i = i - len(self.fut_dates)
        v = 0
        for tenor in range(1, self.swap_tenors[i] + 1):
            
            log_df = cs_func(tenor)
            df = np.exp(log_df)
            
            daily_expiries, daily_dt = self._daily_expiry(tenor - 1, tenor)
            daily_log_discounts = cs_func(daily_expiries)
            sofr_fwd = self._log_df_to_fwd(daily_expiries, daily_log_discounts)
            
            flt_increment = ((sofr_fwd * daily_dt + 1).prod() - 1 ) * df

            fix_increment = self.swap_rates[i] * df
            v += fix_increment - flt_increment
                
        return v ** 2 * 10000
   
    def _objective(self, df_i_plus_1, i):
    # index i is assigned value as j-1 acccording to the _iterative_optimize()
    # which means the ith calibration instrument corresponds to i+1 th discount 
    # factor
    
        df = np.append(self.df[:i+1], df_i_plus_1)
        log_discounts = np.log(df)
    
    # df and pillar_expiries have the same lenth of n+1, now that df ends 
    # at i + 2, the pillar_expiries array also ends at i+2, the interpolation 
    # function cs_func is defined and will be passed to the SOFR swap valuation 
    # and SOFR futures valuation
    
        if self.if_linear:
            cs_func = interp1d(self.pillar_expiries[:i+2], log_discounts,
                                  fill_value="extrapolate")
        else:
            cs_func = CubicSpline(self.pillar_expiries[:i+2], log_discounts,
                                   bc_type = 'natural')
        if i < len(self.fut_dates):
            return self.fut_value_diff(i, cs_func)
        
        return self.swap_value_diff(i, cs_func)       
        
    # Optimization function, looping from the 2nd to the last element of discount 
    # factor array-df, as the first element of the array is "1", the total number 
    # of elements needs to be solved is len(df)-1 which is n
    
    def _optimize(self):
   
        for j in range(1, len(self.df)):
            self._iterative_optimize(j)        
        return
    
    # j-th item in self.df corresponds to the j-th bootstrapping instrument
    
    def _iterative_optimize(self, j):
        bnds = [(0.15, 1)]
     
        self.results = minimize(self._objective, self.df[j - 1], 
                   args=(j - 1), method="SLSQP", bounds=bnds, 
                   options={'disp': False}
                  )
      
        self.df[j] = self.results.x
        return
    
 # Reading the bootstrapping instruments
  
sofr_data = pd.read_csv("rzd_rates.csv", index_col=0)
sofr_realized, sofr_rlz_dates = (sofr_data/100).values.reshape((len(sofr_data, ))),\
                                 sofr_data.index

sr3_data = pd.read_csv("3m_futures_data.csv", header=0, index_col=0)

fut_quotes = sr3_data.iloc[0:7,4].values

fut_dates = sr3_data.iloc[0:7,3].values

swap_tenors = np.array([24, 36, 48, 60, 72, 84, 96, 108, 120, 144, 
                        180, 240, 360, 480]) // 12
swap_rates = np.array([0.000460, 0.000680, 0.001020, 0.001550, 0.002110
                       , 0.002610, 0.003090, 0.003510, 0.003840
                       , 0.004410, 0.004930, 0.005430, 0.005660, 0.005210
                       ])

# Call the class for specific instance and plot the bootstrapping results

SOFR_raw = sofr_curve('04-30-2020', sofr_rlz_dates, sofr_realized, 
           fut_dates, fut_quotes, swap_tenors, swap_rates, True)

SOFR_raw.plot_fwd()

SOFR_cubic = sofr_curve('04-30-2020', sofr_rlz_dates, sofr_realized, 
           fut_dates, fut_quotes, swap_tenors, swap_rates, False)

SOFR_cubic.plot_fwd()

