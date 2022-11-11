#import bamboolib as bam
import os
import os.path

import numpy as np
from yahoo_fin import stock_info as si
from datetime import datetime, timedelta
from VBTT2_IO.IO import write_list, read_list,delete_then_get_model_from_bucket,read_config_file
import time



#functions to fetch and process SP500 data

def read_create_write_SP500(SP500_tickers,filename_json):
    # This create the complete master list of all tickers in SP500, their sectors, their industries
    # Check if SP500 json file exist first otherwise process SP500

    check=delete_then_get_model_from_bucket(filename_json) #delete local file and copy file from bucket to have fresh one
    #file_exists = os.path.exists(filename_json) #no longer required with blob check above

    if check==True:
        SP500_list = read_list(filename_json)
        SP500_list = np.array(SP500_list)  # we need an array
    else:
        SP500_list = read_write_SP500(SP500_tickers,filename_json)  # this will extract

    return SP500_list


def get_ticker_sector(ticker):
    # version 2
    # prerequisites is to import Yahoo_finance.stock_info
    # probleme avec cette methode c'est que industry est ligne 18 ou 19 *
    # prerequisites 2 is to import pandas as pd; import numpy as np
    filename_json="SP500.json"
    file_exists = os.path.exists(filename_json)
    if file_exists:  # file json exist
        SP500_list = read_list(filename_json)
        SP500_list = np.array(SP500_list)  # we need an array
        #now we should extract the sector which is column 2 for in each item of this 2D array
        ticker_sector=SP500_list[:, 1:2][SP500_list[0:, 0:1] == ticker].tolist()

        return ticker_sector[0]
    else:
        df = si.get_company_info(ticker)  # a utiliser pour trouver le secteur
        df = df.reset_index()
        sector = df.loc[df['Breakdown'].isin(['sector'])]  # from bamboolib to extract sector
        sector = sector.iloc[0, 1]  # this is to extract just the value
        industry = df.loc[df['Breakdown'].isin(['industry'])]
        industry = industry.iloc[0, 1]
        ticker_sector = []
        ticker_sector.append([ticker, sector, industry])
        return ticker_sector[0]


def read_write_SP500(tickers_list,filename_json):
    # Initialisation of SP500 data - find sector, industry for all tickers in SP500

    SP500_list = []
    for ticker in tickers_list:
        print(ticker)
        SP500_list.append(get_ticker_sector(ticker))
        if len(SP500_list) % 30 == 0:
            time.sleep(30)
            # for some reason processing SP500 one shot is failing. so sleep of 20 seconds for each 50 tickers
    SP500_list = np.array(SP500_list)
    # Save in a file to reduce processing time next time
    write_list(SP500_list.tolist(), filename_json)  # this function work if it is a list
    return SP500_list  # this returns the array, not the list


def get_all_tickers_sector(sector):
    # prerequisites list is an array of ticker, sector and industry
    filename_json = "SP500.json"
    SP500_list = read_list(filename_json)
    SP500_list = np.array(SP500_list)  # we need an array


    sub_list_tickers = np.array(SP500_list)
    fltr = np.asarray([sector])
    result = sub_list_tickers[np.in1d(sub_list_tickers[:, 1], fltr)]
    return result[:, 0:1]
    # on veut extraire les tickers donc toutes les rangees (0:0) et la colonne 0 donc (0:1)


def get_all_tickers_industry(industry):
    # prerequisites list is an array of ticker, sector and industry
    filename_json = "SP500.json"
    SP500_list = read_list(filename_json)
    SP500_list = np.array(SP500_list)  # we need an array

    sub_list_tickers = np.array(SP500_list)
    fltr = np.asarray([industry])
    result = sub_list_tickers[np.in1d(sub_list_tickers[:, 2], fltr)]
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
    if date_predict.hour>=17:
        date_predict=date_predict+timedelta(1)
    return date_predict