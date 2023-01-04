#import bamboolib as bam
import os
import os.path

import numpy as np
from yahoo_fin import stock_info as si
from datetime import datetime, timedelta
from VBTT2_IO.IO import write_list, read_list,delete_then_get_model_from_bucket,read_config_file,instantiate_logging
import time
import yfinance as yf  #this is not yahoo_fin



#functions to fetch and process SP500 data

def read_create_write_SP500(SP500_tickers,filename_json):
    logger=instantiate_logging()
    # This create the complete master list of all tickers in SP500, their sectors, their industries
    # Check if SP500 json file exist first otherwise process SP500

    check=delete_then_get_model_from_bucket(filename_json) #delete local file and copy file from bucket to have fresh one
    #file_exists = os.path.exists(filename_json) #no longer required with blob check above

    if check==True:
        SP500_list = read_list(filename_json)
        SP500_list = np.array(SP500_list)  # we need an array
    else:
        logger.log_text("Function read_create_write_SP500|result of function delete_then_get_model_from_bucket is false| calling function read_write_SP500")
        SP500_list = read_write_SP500(SP500_tickers,filename_json)  # this will extract

    return SP500_list


def get_ticker_sector(ticker,filename_json):
    # version 2
    logger=instantiate_logging()

    
    # prerequisites is to import Yahoo_finance.stock_info
    # probleme avec cette methode c'est que industry est ligne 18 ou 19 *
    # prerequisites 2 is to import pandas, as pd; import numpy as np
    file_exists = os.path.exists(filename_json)
    if file_exists:  # file json exist
        SP500_list = read_list(filename_json)
        SP500_list = np.array(SP500_list)  # we need an array
        #now we should extract the sector which is column 2 for in each item of this 2D array
        ticker_sector=SP500_list[:, 1:2][SP500_list[0:, 0:1] == ticker].tolist()

        #return ticker_sector[0]

        return ticker_sector
        
    else:
    
        try:
        	df = si.get_company_info(ticker)  # a utiliser pour trouver le secteur
        except:
            logger.log_text(f"Function get_ticker_sector|result of si.get_company_info from Yahoo unsuccessful| returning blank ticker {ticker} ")
            ticker_sector = []
            return ticker_sector
        	
        
        else:
            df = df.reset_index()
            sector = df.loc[df['Breakdown'].isin(['sector'])]  # from bamboolib to extract sector
            sector = sector.iloc[0, 1]  # this is to extract just the value
            industry = df.loc[df['Breakdown'].isin(['industry'])]
            industry = industry.iloc[0, 1]
            ticker_sector = []
            ticker_sector.append([ticker, sector, industry])
            Logger.log_text(f"Function get_ticker_sector|result of si.get_company_info from Yahoo successful| returning info {ticker_sector[0]}")
            return ticker_sector[0]
        


def read_write_SP500(tickers_list,filename_json):
    # Initialisation of SP500 data - find sector, industry for all tickers in SP500
    logger=instantiate_logging()
    SP500_list = []
    for ticker in tickers_list:

        SP500_list.append(get_ticker_sector(ticker,filename_json))
        if len(SP500_list) % 30 == 0:
            time.sleep(1)
            # for some reason processing SP500 one shot is failing. so sleep of 20 seconds for each 50 tickers
    SP500_list = np.array(SP500_list)
    # Save in a file to reduce processing time next time
    write_list(SP500_list.tolist(), filename_json)  # this function work if it is a list
    logger.log_text(f"Function read_write_SP500|List of tickers written in {filename_json}  as follow: {SP500_list.tolist()}.")
    return SP500_list  # this returns the array, not the list


def get_all_tickers_sector(sector):
    # prerequisites list is an array of ticker, sector and industry
    logger=instantiate_logging()
    filename_json = "SP500.json"
    SP500_list = read_list(filename_json)
    SP500_list = np.array(SP500_list)  # we need an array


    sub_list_tickers = np.array(SP500_list)
    fltr = np.asarray([sector])
    result = sub_list_tickers[np.in1d(sub_list_tickers[:, 1], fltr)]
    logger.log_text(f"Function get_all_tickers_sector|List of tickers for {sector} was generated as follow: {result[:, 0]}.")
    return result[:, 0]
    # on veut extraire les tickers donc toutes les rangees (0:0) et la colonne 0 donc (0:1)


def get_all_tickers_industry(industry):
    # prerequisites list is an array of ticker, sector and industry
    logger=instantiate_logging()
    filename_json = read_config_file()[5]
    SP500_list = read_list(filename_json)
    SP500_list = np.array(SP500_list)  # we need an array

    sub_list_tickers = np.array(SP500_list)
    fltr = np.asarray([industry])
    result = sub_list_tickers[np.in1d(sub_list_tickers[:, 2], fltr)]
    logger.log_text(f"Function get_all_tickers_industry|List of tickers for  {industry} was generated as follow: {result[:, 0:1]}.")
    return result[:, 0:1]
    # on veut extraire les tickers donc toutes les rangees (0:0) et la colonne 0 donc (0:1)


def generate_enhanced_data(sector,ticker):
    additional_data = read_config_file()[1]
    sector_list=get_all_tickers_sector(sector)
    additional_data = np.concatenate((sector_list, additional_data), axis=None)
    additional_data = additional_data.tolist()
    additional_data.remove(ticker) #to remove ticker from additional data because we already have it somewhere else
    return additional_data

def YF_datetime():
    date_predict=datetime.now()
    if date_predict.hour>=16:
        date_predict=date_predict+timedelta(1)
    return date_predict
    
    
def get_SP500(filename_json):
    logger=instantiate_logging()
    #generate  sp500 file that contains sector and industry
    SP500_tickers = si.tickers_sp500()
    #exemple - SP500_tickers=['A', 'ETR', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABMD']
    SP500_list = yfinance_read_write_SP500(SP500_tickers, filename_json)
    logger.log_text(f"Function get_sp500|JSON {filename_json} was generated  successfully.")

    return f"JSON {filename_json} was generated  successfully."



def yfinance_read_write_SP500(tickers_list,filename_json):
    logger=instantiate_logging()
    SP500_list = []
    for ticker in tickers_list:

        ticker_company_info = yf.Ticker(ticker).info
        try:
            sector = ticker_company_info['sector']
        except:
            logger.log_text(f"Function yfinance_read_write_SP500|processed failed no sector, industry-> {ticker}.")
        else:
            sector = ticker_company_info['sector']
            industry = ticker_company_info['industry']
            SP500_list.append([ticker, sector, industry])
            logger.log_text(f"Function yfinance_read_write_SP500|processed success-> [{ticker},{sector},{industry}] into SP500_list.")
    SP500_list = np.array(SP500_list)
    # Save in a file to reduce processing time next time
    write_list(SP500_list.tolist(), filename_json)  # this function work if it is a list
    logger.log_text(f"Function read_write_SP500|List of tickers written in {filename_json}  as follow: {SP500_list.tolist()}.")
    return SP500_list



