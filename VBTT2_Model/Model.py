import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from yahoo_fin import stock_info as si
import joblib

from VBTT2_Features.Features import initialize_data, preprocessing,create_train_test_set
from VBTT2_SP500.SP500 import read_create_write_SP500, generate_enhanced_data,get_ticker_sector
from VBTT2_IO.IO import read_config_file,upload_file_to_bucket,instantiate_logging


def select_regressor(MODEL):
    #     # import the regressor
    logger=instantiate_logging()

    if MODEL == 'DecisionTree()':
        from sklearn.tree import DecisionTreeRegressor
        MODEL = DecisionTreeRegressor()
    elif MODEL == 'LinearRegression()':
        from sklearn.linear_model import LinearRegression
        MODEL = LinearRegression()
    elif MODEL == 'svm.SVR()':
        from sklearn import svm
        MODEL = svm.SVR()
    elif MODEL == "DecisionTree(max_depth=5)":
        from sklearn.tree import DecisionTreeRegressor
        MODEL = DecisionTreeRegressor(max_depth=5)
    elif MODEL == "Ridge(alpha=1.0)":
        from sklearn.linear_model import Ridge
        MODEL = Ridge(alpha=1.0)
    elif MODEL == "Lasso(alpha=1.0)":
        from sklearn.linear_model import Lasso
        MODEL = Lasso(alpha=1.0)
    elif MODEL == "xgboost":
        import xgboost as xgb
        MODEL = xgb.XGBClassifier()
    elif MODEL == "RandomForestRegressor(n_estimators=100)":
        from sklearn.ensemble import RandomForestRegressor
        MODEL = RandomForestRegressor(n_estimators=100)
    else:
        # MODEL=='LinearRegression'
        from sklearn.linear_model import LinearRegression
        MODEL = LinearRegression()

    logger.log_text(f"Function select_regressor|  Regressor from config file:  {MODEL}")

    return MODEL



def Model_Train_Save(ticker_list_for_models, years, lags, additional_data, nb_predict_days):
    # Functions for model training and algorythm data and import
    logger=instantiate_logging()

    version, additional_data, regressor, model, bucket, filename_json = read_config_file()
    MODEL=select_regressor(model)




    #### Initialisation of variables and data
    # the timeframe for training and test sets and predict
    days = 360 * years  # Nunber of days in the model
    yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last, days = initialize_data(
        days,
        lags,
        nb_predict_days)
    SP500_tickers = si.tickers_sp500()  # get list of tickers

    # print(f"SP500_tickers = {SP500_tickers}")



    # read master list of ticker, sector, industries or if not found, create it and save it#
    filename_json=read_config_file()[5]
    SP500_list = read_create_write_SP500(SP500_tickers, filename_json)

    ### validation if we have everything


    logger.log_text(f"Function Model_train_Save|  Input ->ticker list {ticker_list_for_models}")
    logger.log_text(f"Function Model_train_Save|  Input ->years {years}")
    logger.log_text(f"Function Model_train_Save|  Input->lags {lags}")
    logger.log_text(f"Function Model_train_Save|  Input ->predict days {nb_predict_days}")
    logger.log_text(f"Function Model_train_Save|  Period->yesterday {yesterday}")
    logger.log_text(f"Function Model_train_Save|  Period->train_date_start {train_date_start}")
    logger.log_text(f"Function Model_train_Save|  Period->train_date_last {train_date_last}")
    logger.log_text(f"Function Model_train_Save|  Period->test_date_start {test_date_start}")
    logger.log_text(f"Function Model_train_Save|  Period->test_date_last {test_date_last}")

    logger.log_text(f"Function Model_train_Save| Tickers for our model {ticker_list_for_models}")
    logger.log_text(f"Function Model_train_Save| Additional data  we add to the tickers for the model {additional_data}")

    Logger.log_text(f"Function Model_train_Save| Model VALIDATE - This is the number of training days of the train dataset {days}")

    # Get features
    #matrix_features_sector = preprocessing(ticker_list_for_models, additional_data, days)

    #### Run models for all tickers selected in input  and predict

    predictions = pd.DataFrame()  # to store predictions

    filename_json="SP500.json"
    for ticker in ticker_list_for_models:
        sector = get_ticker_sector(ticker,filename_json)
        #additional_data = read_config_file()[1]
        additional_data = generate_enhanced_data(sector,ticker)


        matrix_features_sector = preprocessing(ticker, additional_data, days)

        X_train, y_train, X_test, y_test, df_filtered = create_train_test_set(ticker, matrix_features_sector, lags,
                                                                              additional_data, days, nb_predict_days)

        MODEL.fit(X_train, y_train)
        # save the model to disk
        filename = ticker + '_model.sav'
        joblib.dump(MODEL, filename)
        upload_file_to_bucket(filename)
        temp_pred = model_predict(MODEL, ticker, X_test, y_test)

        predictions = predictions.append(temp_pred, ignore_index=True)  # this is to store in the master pandas list

    # add binary buy=1 and sell=0
    df_lagged = add_buy_sell_to_prediction(predictions,ticker_list_for_models)
    df_lagged  # lag_lagged is a DF containing predictions + buy and Sell label

    ticker = "*all*"
    accuracy = balanced_accuracy(ticker, df_lagged)
    #print(f"Accuracy score for {ticker} is {accuracy}.")

    accuracy = []
    for ticker in ticker_list_for_models:
        accuracy.append([balanced_accuracy(ticker, df_lagged), ticker])
    DF_accuracy = pd.DataFrame(accuracy, columns=["Blc accuracy", "Ticker"])
    DF_accuracy


def balanced_accuracy(ticker, predict):
    # put *all* to have global accuracy score for all the predictions
    # or put a ticker name
    if ticker == '*all*':
        return balanced_accuracy_score(predict['y_testb'], predict['y_predb'])
    else:
        return balanced_accuracy_score(predict['y_testb'][predict['ticker'] == ticker],
                                       predict['y_predb'][predict['ticker'] == ticker])


def model_predict(MODEL, ticker, X, y):
    y_pred = MODEL.predict(X)
    temp_pred = predictions_compile(y, y_pred, ticker)  # this is to store temporary ytest, ypredict, ticker
    return temp_pred


def predictions_compile(y_test, y_pred, ticker):
    # this allow to create a dataframe of y_test, y_predict for a given ticker
    predict_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    predict_df['ticker'] = ticker
    return predict_df
    # we need to have an initial empty dataframe to store the predictions



def add_buy_sell_to_prediction(predictions, ticker_list_for_models):
    # this section is to calculate the label=buy or sell
    # it is adding colum y_testb and y_predictb to data frame predictions
    # buy=1
    # sell=0

    df_lagged = pd.DataFrame()
    for ticker in ticker_list_for_models:
        df = predictions[predictions['ticker'] == ticker]
        df_lagged_ticker = df.copy()
        for window in range(1, 1 + 1):
            shifted = df.shift(window)
            shifted.columns = [x + "_lag" + str(window) for x in df.columns]

            df_lagged_ticker = pd.concat((df_lagged_ticker, shifted), axis=1)
            df_lagged_ticker = df_lagged_ticker.dropna()
            df_lagged_ticker['y_testb'] = np.floor(df_lagged_ticker['y_test'] / df_lagged_ticker['y_test_lag1']).astype(
                int)
            df_lagged_ticker['y_predb'] = np.floor(df_lagged_ticker['y_pred'] / df_lagged_ticker['y_pred_lag1']).astype(
                int)
            # *** using map function to decide buy or sell
            category = {1: "Buy", 0: "Sell"}
            df_lagged_ticker['y_recommend'] = df_lagged_ticker['y_predb'].map(category)
            df_lagged_ticker['daily return %'] = (
                        (df_lagged_ticker['y_test'] / df_lagged_ticker['y_test_lag1']) - 1).where \
                (df_lagged_ticker['y_predb'] == 1, \
                 (((df_lagged_ticker['y_test_lag1'] / df_lagged_ticker['y_test']) - 1)))
            df_lagged_ticker['daily return %'] = df_lagged_ticker['daily return %'] * 100
        df_lagged = df_lagged.append(df_lagged_ticker)

    return df_lagged
