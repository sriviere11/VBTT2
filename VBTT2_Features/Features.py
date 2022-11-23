
#functions to deal with creating and extracting features

#import bamboolib as bam

# import bamboolib as bam
# import bamboolib as bam
from datetime import datetime, timedelta
from VBTT2_IO.IO import  read_config_file
from VBTT2_SP500.SP500 import YF_datetime
# import bamboolib as bam
import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si



def get_yf_dataframe(data, nbdays):
    #this function return a dataframe of ticker data for last nbdays
    yesterday = datetime.now()  # we want data up to yesterday
    start_date = yesterday - timedelta(days=nbdays)  # we run the model using data for nbdays
    df_res = pd.DataFrame()
    print("module get_yf_dataframe - data", data)
    print("module get_yf_dataframe - nbdays", nbdays)
    for ticker in data:
        df_tmp = si.get_data(ticker, start_date, yesterday)
        df_res[ticker] = df_tmp['close']
    return df_res



def preprocessing(ticker, additional_data, days):
    #this function return a matrix of features augmented with fix data for number of days
    #this function will cal get_yf_dataframe to actually import the data from Yahoo
    print ("module preprocing - ticker",ticker)
    print("module preprocing - additional_data",additional_data)
    print("module preprocing - nbdays", days)
    tickers_in_sector_extended = np.concatenate((ticker, additional_data), axis=None)
    tickers_in_sector_extended = tickers_in_sector_extended.tolist()
    matrix_features_sector = get_yf_dataframe(tickers_in_sector_extended, days)
    matrix_features_sector = matrix_features_sector.reindex(sorted(matrix_features_sector.columns), axis=1)

    return matrix_features_sector


def create_train_test_set(ticker, features, lags,additional_data,days,nb_predict_days):
    # creating train and test set
    yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last, days = initialize_data(
        days,
        lags,
        nb_predict_days)

    df = features[[ticker] + additional_data]
    df_lagged=pd.DataFrame()
    df_ticker=df[[ticker]]
    for window in range(1, lags + 1):
        shifted = df.shift(window)
        shifted.columns = [x + "_lag" + str(window).zfill(3) for x in df.columns]  #zfill prepad a string with 0

        df_lagged = pd.concat((df_lagged, shifted), axis=1)

    df_lagged= df_lagged.interpolate(method='linear')   #to replace the na on some of the series
    df_lagged=pd.concat((df_ticker,df_lagged),axis=1)
    df_lagged = df_lagged.dropna()
    df_lagged = df_lagged.reindex(sorted(df_lagged.columns), axis=1)


    # train_set
    df_filtered = df_lagged.loc[:train_date_last]
    X_train = df_filtered.drop(columns=[ticker])
    y_train = df_filtered[ticker]

    # test set
    df_filtered = df_lagged.loc[test_date_start:test_date_last]
    X_test = df_filtered.drop(columns=[ticker])
    y_test = df_filtered[ticker]

    # we convert to numpy array
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, y_train, X_test, y_test, df_filtered


def initialize_data(days, lags, nb_predict_days):
    # define main date in models for defining training and test sets dates.
    from datetime import datetime, timedelta

    yesterday=datetime.now()
    start_date = yesterday - timedelta(days=days)
    train_date_start = start_date.strftime("%Y-%m-%d")
    #train_date_last=datetime(2022, 8, 31)
    train_date_last = yesterday - timedelta(days=nb_predict_days + 1)  # nombre de jours a predire
    train_date_last = train_date_last.strftime("%Y-%m-%d")

    test_date_start = yesterday - timedelta(days=nb_predict_days)
    test_date_start = test_date_start.strftime("%Y-%m-%d")
    test_date_last = yesterday.strftime("%Y-%m-%d")

    return yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last,days




def create_predict_set(ticker, features, lags, nb_predict_days, additional_data):
    yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last, days = initialize_data(
        lags * 2, lags, nb_predict_days)
    #  List of features X_train, y_train, X_test,y_test

    df = features[[ticker] + additional_data]

    # duplicate last row of features to use in the lag - this is to add tomorrow date in the predict set
    df_last = df.iloc[-1:]  # we get last date
    # last_date=datetime.fromtimestamp(df_last.index[-1]) #index is of type timestamp therefore convert to datetime
    last_date = df_last.index[-1]  # index is of type timestamp therefore convert to datetime
    df = df.append(df_last)  # we append last date to end
    # df.set_axis((df.index[:-1].union([last_date],sort=False)),axis=0)


    # - Note this does not work especially during weekend. better use below

    date_predict = YF_datetime()
    date_predict = date_predict.strftime("%Y/%m/%d")
    df.index.array[-1] = date_predict
    # df.index.array[-1] = last_date+timedelta(1) #we change to tomorrows date

    #df_lagged = df.copy()
    df_lagged=pd.DataFrame()
    df_ticker=df[[ticker]]
    for window in range(1, lags + 1):
        shifted = df.shift(window)
        shifted.columns = [x + "_lag" + str(window).zfill(3) for x in df.columns]

        df_lagged = pd.concat((df_lagged, shifted), axis=1)

    # df_lagged = df_lagged.fillna(method='ffill')
    df_lagged = df_lagged.interpolate(method='linear')
    df_lagged=pd.concat((df_ticker,df_lagged),axis=1)
    df_lagged = df_lagged.dropna()
    df_lagged = df_lagged.reindex(sorted(df_lagged.columns), axis=1)
    df_lagged.to_csv('df_lagged1.csv')

    # test set
    # df_filtered = df_lagged.loc[test_date_start:test_date_last]
    df_filtered = df_lagged



    X_test = df_filtered.drop(columns=[ticker])
    y_test = df_filtered[ticker]

    # we convert to numpy array
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    return X_test, y_test, df_filtered




