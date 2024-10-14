# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 21:48:37 2024

@author: Diego
"""

import os
import pandas as pd

class BoxSpreadData:
    
    def __init__(self):
        
        self.link = r"https://williamdiamond.weebly.com/uploads/1/4/3/8/143847793/box_gov_07302019.xlsx"
        
        self.tsy_tickers = ["TU", "TY", "UXY", "WN", "FV", "US"]
        self.tsy_path = r"C:\Users\Diego\Desktop\app_prod\BBGFuturesManager\data\PXFront"
        self.deliv_path = r"C:\Users\Diego\Desktop\app_prod\BBGFuturesManager\data\BondDeliverableRisk"
        
        self.parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.data_path = os.path.join(self.parent_path, "data")
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        
        
    def get_box_spread(self, verbose: bool = False):
        
        file_path = os.path.join(self.data_path, "BoxSpread.parquet")

        try:
            
            if verbose == True: print("Looking for data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't Find Data Collecting it")
            
            df_out = (pd.read_excel(
                io = self.link).
                assign(
                    str_date = lambda x: x.year.astype(str) + "-" + x.month.astype(str) + "-" + x.day.astype(str),
                    date = lambda x: pd.to_datetime(x.str_date, format = "%Y-%m-%d").dt.date).
                drop(columns = ["str_date", "year", "month", "day"]).
                melt(id_vars = "date").
                assign(
                    tenor = lambda x: x.variable.str.split("_").str[1],
                    rate = lambda x: x.variable.str.split("_").str[0]).
                drop(columns = ["variable"]).
                pivot(index = ["date", "tenor"], columns = "rate", values = "value").
                reset_index().
                assign(spread = lambda x: x.box - x.gov))
            
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
            
    def _get_rtn(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(
                date = lambda x: pd.to_datetime(x.date).dt.date,
                security = lambda x: x.security.str.split(" ").str[0],
                PX_RTN = lambda x: x.PX_LAST.pct_change(),
                PX_BPS = lambda x: x.PX_LAST.diff() / x.duration))
        
        return df_out
        
    def get_tsy_fut(self, verbose: bool = False): 
        
        file_path = os.path.join(self.data_path, "TSYFutures.parquet")
        
        try:
            
            if verbose == True: print("Looking for data")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't Find Data Collecting It")
            fut_paths = [os.path.join(self.tsy_path, ticker + ".parquet") for ticker in self.tsy_tickers]
            df_fut = (pd.read_parquet(
                path = fut_paths, engine = "pyarrow"))
            
            deliv_paths = [os.path.join(self.deliv_path, ticker + ".parquet") for ticker in self.tsy_tickers]
            df_deliv = (pd.read_parquet(
                path = deliv_paths, engine = "pyarrow").
                query("variable == 'CONVENTIONAL_CTD_FORWARD_FRSK'").
                drop(columns = ["variable"]).
                rename(columns = {"value": "duration"}))
    
            df_out = (df_fut.merge(
                right = df_deliv, how = "inner", on = ["date", "security"]).
                groupby("security").
                apply(self._get_rtn).
                reset_index(drop = True).
                dropna())
            
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
        
def main():
    
    box_spread_data = BoxSpreadData()
    box_spread_data.get_box_spread(verbose = True)
    box_spread_data.get_tsy_fut(verbose = True)
    
#if __name__ == "__main__": main()
    
