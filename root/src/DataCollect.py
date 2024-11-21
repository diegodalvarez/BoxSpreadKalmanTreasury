# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:46:20 2024

@author: Diego
"""

import os
import pandas as pd

class BoxSpreadData:
    
    def __init__(self):
        
        self.link = r"https://williamdiamond.weebly.com/uploads/1/4/3/8/143847793/box_gov_07302019.xlsx"
        self.intl_link = r"https://williamdiamond.weebly.com/uploads/1/4/3/8/143847793/data_public_daily.xlsx"
        
        self.tsy_tickers = ["TU", "TY", "UXY", "WN", "FV", "US"]
        self.tsy_path = r"C:\Users\Diego\Desktop\app_prod\BBGFuturesManager\data\PXFront"
        self.deliv_path = r"C:\Users\Diego\Desktop\app_prod\BBGFuturesManager\data\BondDeliverableRisk"
        self.bbg_path = r"C:\Users\Diego\Desktop\app_prod\BBGData\data"
        
        self.root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.repo_path = os.path.abspath(os.path.join(self.root_path, os.pardir))
        self.data_path = os.path.join(self.repo_path, "data")
        self.raw_path  = os.path.join(self.data_path, "RawData")
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_path)  == False: os.makedirs(self.raw_path)
        
        self.misc_tickers = ["GVLQUSD"]
        
    def get_box_spread(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.raw_path, "BoxSpread.parquet")

        try:
            
            if verbose == True: print("Looking for Box Spread data")
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
    
    def _get_intl_box_spread(self, sheet_name: str, verbose: bool = False) -> pd.DataFrame: 
        
        if verbose == True: print("Working on {}".format(sheet_name))
        df_tmp = (pd.read_excel(
            io = self.intl_link, sheet_name = sheet_name).assign(
            str_date = lambda x: x.date_year.astype(str) + "-" + x.date_month.astype(str) + "-" + x.date_day.astype(str),
            date     = lambda x: pd.to_datetime(x.str_date, format = "%Y-%m-%d")).
            drop(columns = ["date_year", "date_month", "date_day", "str_date"]).
            melt(id_vars = "date").
            assign(
                country = sheet_name.replace(" ", "_").lower(),
                tenor   = lambda x: x.variable.str.split("_").str[1],
                rate    = lambda x: x.variable.str.split("_").str[0]).
            drop(columns = ["variable"]))
        
        return df_tmp
    
    def get_intl_box_spread(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "IntlBoxSpread.parquet")
        
        try: 
            
            if verbose == True: print("Looking for international")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find Data, Collecting it")
            sheet_names = ["United States", "Europe", "Switzerland", "United Kingdom"]
            
            df_out = (pd.concat([
                self._get_intl_box_spread(sheet_name, verbose)
                for sheet_name in sheet_names]))
            
            print("Saving Data")
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
        
    def get_tsy_fut(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "TSYFutures.parquet")
        
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
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def get_misc_indices(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_path, "MiscIndices.parquet")
        try:
            
            if verbose == True: print("Looking for Misc Indices")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't Find Data Collecting It")
            paths = [
                os.path.join(self.bbg_path, ticker + ".parquet")
                for ticker in self.misc_tickers]
            
            df_out = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                drop(columns = ["variable"]).
                assign(
                    date     = lambda x: pd.to_datetime(x.date).dt.date,
                    security = lambda x: x.security.str.split(" ").str[0]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out

def main() -> None: 
        
    BoxSpreadData().get_box_spread(verbose = True)
    BoxSpreadData().get_intl_box_spread(verbose = True)
    BoxSpreadData().get_tsy_fut(verbose = True)
    BoxSpreadData().get_misc_indices(verbose = True)
#if __name__ == "__main__": main()