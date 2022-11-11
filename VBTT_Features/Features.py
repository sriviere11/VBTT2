
#functions to deal with creating and extracting features

from datetime import datetime, timedelta
from yahoo_fin import stock_info as si
import numpy as np


# import bamboolib as bam
import joblib
import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si


def get_yf_dataframe(data, nbdays):
    yesterday = datetime.now() - timedelta(1)  # we want data up to yesterday
    start_date = yesterday - timedelta(days=nbdays)  # we run the model using data for nbdays
    df_res = pd.DataFrame()
    for ticker in data:
        df_tmp = si.get_data(ticker, start_date, yesterday)
        df_res[ticker] = df_tmp['close']
    return df_res


def preprocessing(ticker_list_for_models, additional_data, days):
    # this function return a matrix of features augmented with fix data for number of days

    tickers_in_sector_extended = np.concatenate((ticker_list_for_models, additional_data), axis=None)
    tickers_in_sector_extended = tickers_in_sector_extended.tolist()
    matrix_features_sector = get_yf_dataframe(tickers_in_sector_extended, days)
    # import pandas as pd; import numpy as np
    # matrix_features_sector = matrix_features_sector.reset_index()
    matrix_features_sector = matrix_features_sector.reindex(sorted(matrix_features_sector.columns), axis=1)

    ###################################
    ##### saving and reading features - can help increase processing time
    ##### to evaluate on future version
    ####################################
    # matrix_features_sector.to_csv("matrix_features_sector.csv")
    # matrix_features_sector=pd.read_csv("matrix_features_sector.csv")

    return matrix_features_sector


def create_train_test_set(ticker, features, lags,additional_data,days,nb_predict_days):
    # creating train and test set
    yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last, days = initialize_data(
        days,
        lags,
        nb_predict_days)

    df = features[[ticker] + additional_data]
    df_lagged = df.copy()
    for window in range(1, lags + 1):
        shifted = df.shift(window)
        shifted.columns = [x + "_lag" + str(window) for x in df.columns]

        df_lagged = pd.concat((df_lagged, shifted), axis=1)
    df_lagged = df_lagged.dropna()
    df_lagged = df_lagged.reindex(sorted(df_lagged.columns), axis=1)

    # print(df_lagged)
    # df_lagged[ticker+"_2labels"]=np.floor(df_lagged[ticker]/df_lagged[ticker+"_lag1"]).astype(int)

    # train_set
    df_filtered = df_lagged.loc[:train_date_last]
    # X_train=df_filtered.drop(columns=[ticker, ticker+"_2labels"])
    X_train = df_filtered.drop(columns=[ticker])
    # y_train=df_filtered[ticker+"_2labels"]
    y_train = df_filtered[ticker]

    # test set
    df_filtered = df_lagged.loc[test_date_start:test_date_last]
    # X_test=df_filtered.drop(columns=[ticker, ticker+"_2labels"])
    X_test = df_filtered.drop(columns=[ticker])
    # y_test=df_filtered[ticker+"_2labels"]
    y_test = df_filtered[ticker]

    # we convert to numpy array
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, y_train, X_test, y_test, df_filtered


def initialize_data(days, lags, nb_predict_days):

    from datetime import datetime, timedelta



    # define main date in models for defining training and test sets dates.
    #yesterday = datetime.now() - timedelta(1)
    yesterday=datetime.now()
    start_date = yesterday - timedelta(days=days)
    train_date_start = start_date.strftime("%Y-%m-%d")
    train_date_last = yesterday - timedelta(days=nb_predict_days + 1)  # nombre de jours a predire
    train_date_last = train_date_last.strftime("%Y-%m-%d")

    test_date_start = yesterday - timedelta(days=nb_predict_days)
    test_date_start = test_date_start.strftime("%Y-%m-%d")
    test_date_last = yesterday.strftime("%Y-%m-%d")

    return yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last,days



def create_predict_set(ticker ,features ,lags, nb_predict_days, additional_data):

    yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last, days = initialize_data(
        lags * 2, lags, nb_predict_days)
    #  List of features X_train, y_train, X_test,y_test
    df =features[[ticker ] +additional_data]
    df_lagged =df.copy()
    for window in range(1, lags + 1):
        shifted = df.shift(window)
        shifted.columns = [x + "_lag" + str(window) for x in df.columns]

        df_lagged = pd.concat((df_lagged, shifted), axis=1)
    df_lagged = df_lagged.dropna()
    df_lagged =df_lagged.reindex(sorted(df_lagged.columns), axis=1)

    # training set
    df_filtered = df_lagged.loc[test_date_start:test_date_last]
    X_test =df_filtered.drop(columns=[ticker])
    y_test =df_filtered[ticker]

    # we convert to numpy array
    X_test =X_test.to_numpy()
    y_test= y_test.to_numpy()
    return X_test ,y_test ,df_filtered

### Preprocessing - generate matrix of features for sector or industry
def Model_Train_Save(ticker_list_for_models,years,lags,additional_data, nb_predict_days):
    # Functions for model training and algorythm data and import


    # import the regressors
    from sklearn.tree import DecisionTreeRegressor
    # MODEL = linear_model.LinearRegression()
    # MODEL = svm.SVR()
    MODEL = DecisionTreeRegressor()

    # import balanced_accuracy_score

    #### Initialisation of variables and data
    # the timeframe for training and test sets and predict
    days = 360 * years  # Nunber of days in the model
    yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last, days = initialize_data(days,
                                                                                                                lags,
                                                                                                                nb_predict_days)
    SP500_tickers = si.tickers_sp500()  # get list of tickers

    # print(f"SP500_tickers = {SP500_tickers}")

    #additional_data = read_config_file()
    # read master list of ticker, sector, industries or if not found, create it and save it#
    SP500_list = read_create_write_SP500(SP500_tickers, "SP500.json")

    ### validation if we have everything
    #print(f"Input ->years {years}")
    #print(f"Input ->ticker list {ticker_list_for_models}")
    #print(f"Input->lags {lags}")
    #print(f"Input ->predict days {nb_predict_days}")
    #print(f"Period->yesterday {yesterday}")
    #print(f"period->train_date_start {train_date_start}")
    #print(f"Period->train_date_last {train_date_last}")
    #print(f"Period->test_date_start {test_date_start}")
    #print(f"Period->test_date_last {test_date_last}")

    #print(f"This is the tickers for our model {ticker_list_for_models}")
    #print(f"This is the additional data  we add to the tickers for the model {additional_data}")
    #print(f"VALIDATE - This is the number of training days of the train dataset {days}")

    # Get features
    matrix_features_sector = preprocessing(ticker_list_for_models, additional_data, days)

    #### Run models for all tickers selected in input  and predict

    predictions = pd.DataFrame()  # to store predictions

    for ticker in ticker_list_for_models:
        X_train, y_train, X_test, y_test, df_filtered = create_train_test_set(ticker, matrix_features_sector, lags,additional_data,days,nb_predict_days)
        MODEL.fit(X_train, y_train)
        # save the model to disk
        filename = ticker + '_model.sav'
        joblib.dump(MODEL, filename)
        temp_pred = model_predict(MODEL, ticker, X_test, y_test)

        predictions = predictions.append(temp_pred, ignore_index=True)  # this is to store in the master pandas list

    # add binary buy=1 and sell=0
    df_lagged = add_buy_sell_to_prediction(predictions)
    df_lagged  # lag_lagged is a DF containing predictions + buy and Sell label

    ticker = "*all*"
    accuracy = balanced_accuracy(ticker, df_lagged)
    print(f"Accuracy score for {ticker} is {accuracy}.")

    accuracy = []
    for ticker in ticker_list_for_models:
        accuracy.append([balanced_accuracy(ticker, df_lagged), ticker])
    DF_accuracy = pd.DataFrame(accuracy, columns=["Blc accuracy", "Ticker"])
    DF_accuracy





