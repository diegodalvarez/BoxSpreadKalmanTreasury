# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:47:07 2024

@author: Diego
"""

import os
import numpy as np
import pandas as pd


from   DataCollect import BoxSpreadData
from   pykalman import KalmanFilter

class SignalGenerator(BoxSpreadData):
    
    def __init__(self):
        
        super().__init__()
        self.df_box_spread  = self.get_box_spread()
        self.df_intl_spread = self.get_intl_box_spread()
        
        self.lookbacks = [i for i in range(2, 8)]
        self.windows = [
            {"short_window": 2 ** i, "long_window": 2 ** (i + 1)}
            for i in self.lookbacks]
        
        self.parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.data_path   = os.path.join(self.parent_path, "data")
        self.signal_path = os.path.join(self.data_path, "Signals")
        
        if os.path.exists(self.data_path)   == False: os.makedirs(self.data_path)
        if os.path.exists(self.signal_path) == False: os.makedirs(self.signal_path)
        
        
    def _prep_data(self) -> pd.DataFrame: 
        
        df_intl_tmp = (self.df_intl_spread.query(
            "rate == 'box'").
            assign(
                variable    = lambda x: x.country + " " + x.tenor,
                data_source = "intl",
                date        = lambda x: pd.to_datetime(x.date).dt.date).
            drop(columns = ["country", "rate", "tenor"]))
        
        df_box_tmp = (self.df_box_spread.drop(
            columns = ["box", "gov"]).
            rename(columns = {
                "tenor" : "variable",
                "spread": "value"}).
            assign(data_source = "box"))
        
        df_combined = pd.concat([df_intl_tmp, df_box_tmp])
        return df_combined
    
    def _get_trend(self, df: pd.DataFrame, trend_window: dict, d: int = 10) -> pd.DataFrame: 

        df_tmp = (df.sort_values(
            "date").
            assign(
                short_window = trend_window["short_window"],
                long_window  = trend_window["long_window"],
                short_ma     = lambda x: x.value.ewm(span = trend_window["short_window"], adjust = False).mean(),
                long_ma      = lambda x: x.value.ewm(span = trend_window["long_window"],  adjust = False).mean(),
                signal       = lambda x: (x.short_ma - x.long_ma) / (x.value - 1),
                lag_signal   = lambda x: x.signal.shift(),
                decile       = lambda x: pd.qcut(x = x.lag_signal, q = d, labels = ["D{}".format(i + 1) for i in range(d)]),
                lag_decile   = lambda x: x.decile.shift()).
            dropna())
        
        return df_tmp
    
    def _get_all_trend_windows(
            self, 
            df              : pd.DataFrame, 
            windows         : list, 
            clean_window    : int = 250,
            zscore_threshold: int = 6) -> pd.DataFrame: 

        print("Working on {}".format(df.name))
        df_zscore = (df.assign(
            roll_ma  = lambda x: x.value.ewm(span = clean_window, adjust = False).mean(),
            roll_std = lambda x: x.value.ewm(span = clean_window, adjust = False).std(),
            z_score  = lambda x: np.abs((x.value - x.roll_ma) / x.roll_std)).
            query("z_score < 6").
            drop(columns = ["roll_ma", "roll_std", "z_score"]))
    
        df_out = pd.concat([self._get_trend(df_zscore, window) for window in windows])
        return df_out
        
    def get_trend(self, verbose: bool = False) -> pd.DataFrame: 
        
        trend_path = os.path.join(self.signal_path, "TrendSignals.parquet")
        try: 
            
            if verbose == True: print("Trying to find Trend Signals")
            df_out = pd.read_parquet(path = trend_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
        
            if verbose == True: print("Couldn't find it generating signals")
            df_out = (self._prep_data().assign(
                group_var = lambda x: x.variable + " " + x.data_source).
                groupby("group_var").
                apply(self._get_all_trend_windows, self.windows).
                reset_index(drop = True))
            
            if verbose == True: print("Saving Trend Signals\n")
            df_out.to_parquet(path = trend_path, engine = "pyarrow")
        
        return df_out
    
    def _get_resid_zscore(self, df: pd.DataFrame, lookback: int, d: int = 10) -> pd.DataFrame:
        
        df_out = (df.assign(
            lookback   = lookback,
            resid_mean = lambda x: x.resid.ewm(span = lookback, adjust = False).mean(),
            resid_std  = lambda x: x.resid.ewm(span = lookback, adjust = False).std(),
            zscore     = lambda x: (x.resid - x.resid_mean) / x.resid_std,
            lag_zscore = lambda x: x.zscore.shift(),
            decile     = lambda x: pd.qcut(x = x.lag_zscore, q = d, labels = ["D{}".format(i + 1) for i in range(d)]),
            lag_decile = lambda x: x.decile.shift()).
            dropna())
        
        return df_out
    
    def _get_kalman_filter(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame: 
        
        print("Working on {}".format(df.name))
        df = df.dropna()
        kalman_filter = KalmanFilter(
            transition_matrices      = [1],
            observation_matrices     = [1],
            initial_state_mean       = 0,
            initial_state_covariance = 1,
            observation_covariance   = 1,
            transition_covariance    = 0.01)
        
        state_means, state_covariances = kalman_filter.filter(df.value)
        df_kalman = (df.assign(
            smooth     = state_means,
            lag_smooth = lambda x: x.smooth.shift(),
            resid      = lambda x: x.value - x.lag_smooth))
        
        df_out = pd.concat([
            self._get_resid_zscore(df_kalman, lookback)
            for lookback in self.lookbacks])
        
        return df_out
    
    def kalman_filter(self, verbose: bool = False) -> pd.DataFrame: 
        
        kalman_path = os.path.join(self.signal_path, "KalmanSignals.parquet")
        try:
            
            if verbose == True: print("Trying to find Kalman Signals")
            df_out = pd.read_parquet(path = kalman_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it generating signals")
            df_out = (self._prep_data().assign(
                group_var = lambda x: x.variable + " " + x.data_source).
                groupby("group_var").
                apply(self._get_kalman_filter).
                reset_index(drop = True))
            
            if verbose == True: print("Saving Kalman Signals\n")
            df_out.to_parquet(path = kalman_path, engine = "pyarrow")
            
        return df_out

def main():
    
    SignalGenerator().get_trend(verbose = True)    
    SignalGenerator().kalman_filter(verbose = True)
    
if __name__ == "__main__": main()