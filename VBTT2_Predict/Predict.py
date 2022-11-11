# functions to predict model and calculate accuracy

import os
# import bamboolib as bam
import os.path

import joblib
import numpy as np
import pandas as pd

from VBTT2_Features.Features import preprocessing, initialize_data, create_predict_set
from VBTT2_SP500.SP500 import get_ticker_sector, generate_enhanced_data,YF_datetime
from VBTT2_IO.IO import read_config_file,delete_then_get_model_from_bucket
from VBTT2_Model.Model import Model_Train_Save, balanced_accuracy, model_predict, add_buy_sell_to_prediction


'''
def predict_ticker(ticker_list_for_models):
    ticker_list_for_models = ticker_list_for_models.split('-')
    # note when we use flask we will use /ticker/aapl-nflx-cdw
    # so we need to split

    # print(f"Example long_test: {long_test}")
    # print(f"Example short_test: {short_test}")
    # print(f"Example not_working: {not_working}")
    # ticker_list_for_models=input_ticker

    # initialisation model
    # how many years for the model
    years = 6
    lags = 30  # how many days of lags we need of this model, this is like an hyperparameter for us
    version, additional_data, regressor, model,bucket = read_config_file()  # other data needed
    print(version)
    print(regressor)
    print(model)
    print(additional_data)
    print(bucket)

    nb_predict_days = 30  # size of test data in number of days

    # if we don't have model for a ticker in list, retrain model and save
    check = True
    for ticker in ticker_list_for_models:
        check=delete_then_get_model_from_bucket(ticker + "_model.sav") #this download model from bucket. Model will be trained if model does not exist

        if not (os.path.exists(ticker + "_model.sav")):
            check = False
            break  # this allow to continue and not go through the list if file not exist

    if check == False:
        #Retrain all for the select list of tickers
        print(f"Training model for at least one ticker in {ticker_list_for_models}")
        Model_Train_Save(ticker_list_for_models, years, lags, additional_data, nb_predict_days)
    else:
        print(f"no model Training is needed for  {ticker_list_for_models}")
        ## get variable for start the prediction
        ##just in case, we fetch 2 time the lags so that we don't have issue when lagging

    yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last, days = initialize_data(
        lags * 2, lags, nb_predict_days)

    ### validation if we have everything
    print(f"Input ->years {years}")
    print(f"Input ->ticker list {ticker_list_for_models}")
    print(f"Input->lags {lags}")
    print(f"Input ->predict days {nb_predict_days}")
    print(f"Period->yesterday {yesterday}")
    print(f"period->train_date_start {train_date_start}")
    print(f"Period->train_date_last {train_date_last}")
    print(f"Period->test_date_start {test_date_start}")
    print(f"Period->test_date_last {test_date_last}")

    print(f"This is the tickers for our model {ticker_list_for_models}")
    print(f"This is the additional data  we add to the tickers for the model {additional_data}")

    #df_tomorrow = preprocessing(ticker_list_for_models, additional_data, lags * 2)  # add additional features
    #df_tomorrow.shape

    # to store predictions
    Predictions = pd.DataFrame()

    for ticker in ticker_list_for_models:
        # load the saved model for the ticker
        filename = ticker + "_model.sav"
        loaded_model = joblib.load(filename)
        sector=get_ticker_sector(ticker)
        #additional_data=read_config_file()[1]
        additional_data=generate_enhanced_data(sector,ticker)
        print(additional_data)
        df_tomorrow = preprocessing(ticker, additional_data, lags * 2)  # add additional features
        print("predict")
        print(df_tomorrow)

        X_test, y_test, df_filtered = create_predict_set(ticker, df_tomorrow, lags, nb_predict_days,
                                                         additional_data)  # this is X and y

        print(X_test)


        temp_pred = model_predict(loaded_model, ticker, X_test, y_test)
        print('temp_pred')
        print(temp_pred)
        # ===
        temp_pred.drop(labels=[0], inplace=True)

        from datetime import timedelta

        df_filtered2 = pd.DataFrame(df_filtered[ticker])

        df_filtered2 = df_filtered2.reset_index()

        # Deleted 1 row in df_filtered2
        df_filtered2.drop(labels=[0], inplace=True)

        # Renamed columns Date
        df_filtered2.rename(columns={'index': 'Date'}, inplace=True)

        df_filtered2['Predicted for'] = df_filtered2['Date'] + timedelta(days=1)
        # df_filtered2['Prediction for']=df_filtered2['Date']
        df_filtered2 = df_filtered2[['Date', 'Predicted for']]

        # Step: Copy a dataframe column

        temp_pred2 = pd.concat([temp_pred, df_filtered2], axis=1)

        # ===

        Predictions = Predictions.append(temp_pred2, ignore_index=True)  # this is to store in the master pandas list

        # print(f"temp_pred: \n{temp_pred}")
        # print(f"temp_pred2: \n{temp_pred2}")
        # print(f"temp_pred2: \n{temp_pred2}")
        # print(f"df_filtered2: \n{df_filtered2}")

        # print(f"Predictions:\n {Predictions}")

    # adding binary buy or sell to predictions dataframe
    Predictions = add_buy_sell_to_prediction(Predictions,ticker_list_for_models)

    ticker = "*all*"
    accuracy_all= balanced_accuracy(ticker, Predictions)
    print(f"Accuracy score for {ticker} is {accuracy_all}.")
    # provide a data frame of the accuracies

    avg_return = [0]  # render for view html need a first element to be 0
    for ticker in ticker_list_for_models:
        ticker_return = round(Predictions['daily return %'][Predictions['ticker'] == ticker].mean(), 2)
        avg_return.append(ticker_return)

    DF_Recommendations = []
    for ticker in ticker_list_for_models:
        DF_Recommendations.append([ticker, "", "", "", balanced_accuracy(ticker, Predictions)])
        Recommendations = pd.DataFrame(DF_Recommendations,
                                       columns=["Ticker", 'Predicted for', 'Predicted', "Recommended", "Accuracy"])

    # ********************************************************
    # suggestion - DF_accuracy change to DF_accuracy_recommendation
    # suggestion - resultat change as follow: Date, Observed Value, Date Prediction, Predicted Value,Recommendation
    # suggestion - now date observed value should be change to NA
    # ********************************************************

    for ticker in ticker_list_for_models:
        # results = Predictions[['y_test', 'y_pred', 'ticker', 'y_predb','y_recommend']][Predictions['ticker'] == ticker]
        results = Predictions[['y_test', 'y_pred', 'ticker', 'y_predb', 'y_recommend','daily return %']][Predictions['ticker'] == ticker]
        # date_predict=yesterday + timedelta(1)
        date_predict = yesterday
        date_predict = date_predict.strftime("%Y/%m/%d")
        ticker_predicted = results.iloc[-1]['y_pred']  # this is the last row containing result
        ticker_recommend = results.iloc[-1]['y_recommend']  # this is the last row containing result
        Recommendations.loc[Recommendations['Ticker'] == ticker, 'Predicted for'] = date_predict  # to change content of a cell
        Recommendations.loc[Recommendations['Ticker'] == ticker, 'Predicted'] = ticker_predicted
        Recommendations.loc[Recommendations['Ticker'] == ticker, 'Recommended'] = ticker_recommend

        # print(f"Prediction for {yesterday+timedelta(1)} -- ticker: {ticker} {'**resultat**'}\n {resultat.tail()}\n\n")

    Results = Predictions[['ticker', 'Date', 'y_test', 'Predicted for', 'y_pred', 'y_recommend','daily return %']]
    Results = Results.rename(
        columns={'ticker': 'Ticker', 'y_test': 'Observed', 'y_pred': 'Predicted', 'y_recommend': 'Recommended','daily return %':'Daily return %'})

    return Results, Recommendations,avg_return
'''


def predict_ticker(ticker_list_for_models):
    ticker_list_for_models = ticker_list_for_models.split('-')
    # note when we use flask we will use /ticker/aapl-nflx-cdw
    # so we need to split

    # print(f"Example long_test: {long_test}")
    # print(f"Example short_test: {short_test}")
    # print(f"Example not_working: {not_working}")
    # ticker_list_for_models=input_ticker

    # initialisation model
    # how many years for the model
    years = 6
    lags = 30  # how many days of lags we need of this model, this is like an hyperparameter for us
    version, additional_data, regressor, model, bucket = read_config_file()  # other data needed
    print(version)
    print(regressor)
    print(model)
    print(additional_data)
    print(bucket)

    nb_predict_days = 30  # size of test data in number of days

    # if we don't have model for a ticker in list, retrain model and save
    check = True
    for ticker in ticker_list_for_models:
        check = delete_then_get_model_from_bucket(
            ticker + "_model.sav")  # this download model from bucket. Model will be trained if model does not exist

        if not (os.path.exists(ticker + "_model.sav")):
            check = False
            break  # this allow to continue and not go through the list if file not exist

    if check == False:
        # Retrain all for the select list of tickers
        print(f"Training model for at least one ticker in {ticker_list_for_models}")
        Model_Train_Save(ticker_list_for_models, years, lags, additional_data, nb_predict_days)
    else:
        print(f"no model Training is needed for  {ticker_list_for_models}")
        ## get variable for start the prediction
        ##just in case, we fetch 2 time the lags so that we don't have issue when lagging

    yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last, days = initialize_data(
        lags * 2, lags, nb_predict_days)

    ### validation if we have everything
    print(f"Input ->years {years}")
    print(f"Input ->ticker list {ticker_list_for_models}")
    print(f"Input->lags {lags}")
    print(f"Input ->predict days {nb_predict_days}")
    print(f"Period->yesterday {yesterday}")
    print(f"period->train_date_start {train_date_start}")
    print(f"Period->train_date_last {train_date_last}")
    print(f"Period->test_date_start {test_date_start}")
    print(f"Period->test_date_last {test_date_last}")

    print(f"This is the tickers for our model {ticker_list_for_models}")
    print(f"This is the additional data  we add to the tickers for the model {additional_data}")

    # df_tomorrow = preprocessing(ticker_list_for_models, additional_data, lags * 2)  # add additional features
    # df_tomorrow.shape

    # to store predictions
    Predictions = pd.DataFrame()

    for ticker in ticker_list_for_models:
        # load the saved model for the ticker
        filename = ticker + "_model.sav"
        loaded_model = joblib.load(filename)
        sector = get_ticker_sector(ticker)
        additional_data = read_config_file()[1]
        additional_data = generate_enhanced_data(sector, ticker)
        print(additional_data)
        df_tomorrow = preprocessing(ticker, additional_data, lags * 2)  # add additional features
        print("predict df_tomorrow shape before creating predict set")
        print(df_tomorrow.shape)

        X_test, y_test, df_filtered = create_predict_set(ticker, df_tomorrow, lags, nb_predict_days,
                                                         additional_data)  # this is X and y

        print("X_test shape in module predict_ticker before calling model_predict and after creating predict_set",
              X_test.shape)

        temp_pred = model_predict(loaded_model, ticker, X_test, y_test)
        print("temp_pred shape in module predict_ticker after calling model_predict", temp_pred.shape)

        print(temp_pred)
        # ===
        temp_pred.drop(labels=[0], inplace=True)
        temp_pred.to_csv('temp_pred.csv')

        from datetime import timedelta

        df_filtered2 = pd.DataFrame(df_filtered[ticker])

        df_filtered2 = df_filtered2.reset_index()

        # Deleted 1 row in df_filtered2
        df_filtered2.drop(labels=[0], inplace=True)

        # Renamed columns Date
        df_filtered2.rename(columns={'index': 'Date'}, inplace=True)

        df_filtered2['Predicted for'] = df_filtered2['Date']
        # df_filtered2['Prediction for']=df_filtered2['Date']
        df_filtered2 = df_filtered2[['Date', 'Predicted for']]

        # Step: Copy a dataframe column

        temp_pred2 = pd.concat([temp_pred, df_filtered2], axis=1)

        # ===

        Predictions = Predictions.append(temp_pred2, ignore_index=True)  # this is to store in the master pandas list

        # print(f"temp_pred: \n{temp_pred}")
        # print(f"temp_pred2: \n{temp_pred2}")
        # print(f"temp_pred2: \n{temp_pred2}")
        # print(f"df_filtered2: \n{df_filtered2}")

        # print(f"Predictions:\n {Predictions}")

    # adding binary buy or sell to predictions dataframe
    Predictions = add_buy_sell_to_prediction(Predictions, ticker_list_for_models)
    Predictions.to_csv('predictions.csv')

    ticker = "*all*"
    accuracy_all = balanced_accuracy(ticker, Predictions)
    print(f"Accuracy score for {ticker} is {accuracy_all}.")
    # provide a data frame of the accuracies

    avg_return = [0]  # render for view html need a first element to be 0
    for ticker in ticker_list_for_models:
        ticker_return = round(Predictions['daily return %'][Predictions['ticker'] == ticker].iloc[:-1].mean(), 2)
        avg_return.append(ticker_return)

    DF_Recommendations = []
    for ticker in ticker_list_for_models:
        DF_Recommendations.append([ticker, "", "", "", balanced_accuracy(ticker, Predictions.iloc[:-1])])
        Recommendations = pd.DataFrame(DF_Recommendations,
                                       columns=["Ticker", 'Predicted for', 'Predicted', "Recommended", "Accuracy"])

    # ********************************************************
    # suggestion - DF_accuracy change to DF_accuracy_recommendation
    # suggestion - resultat change as follow: Date, Observed Value, Date Prediction, Predicted Value,Recommendation
    # suggestion - now date observed value should be change to NA
    # ********************************************************

    for ticker in ticker_list_for_models:
        # results = Predictions[['y_test', 'y_pred', 'ticker', 'y_predb','y_recommend']][Predictions['ticker'] == ticker]
        results = Predictions[['y_test', 'y_pred', 'ticker', 'y_predb', 'y_recommend', 'daily return %']][
            Predictions['ticker'] == ticker]
        # results['y_test'].iloc[-1]=np.nan  # pas la bonne methode
        results.loc[results.index[-1], 'y_test'] = np.nan
        results['daily return %'].iloc[-1] = np.nan  # pas la bonne methode
        results.loc[results.index[-1], 'daily return %'] = np.nan
        date_predict = YF_datetime() #can be confusing but yesterday is datetime.now. good thing is to take latest in results +1
        date_predict = date_predict.strftime("%Y/%m/%d")
        ticker_predicted = results.iloc[-1]['y_pred']  # this is the last row containing result
        ticker_recommend = results.iloc[-1]['y_recommend']  # this is the last row containing result
        Recommendations.loc[Recommendations['Ticker'] == ticker, 'Predicted for'] = date_predict  # to change content of a cell
        Recommendations.loc[Recommendations['Ticker'] == ticker, 'Predicted'] = ticker_predicted
        Recommendations.loc[Recommendations['Ticker'] == ticker, 'Recommended'] = ticker_recommend
        Predictions.loc[(Predictions['Date'] == date_predict) & (Predictions['ticker'] == ticker), 'y_test'] = np.nan
        Predictions.loc[(Predictions['Date'] == date_predict) & (Predictions['ticker'] == ticker), 'daily return %'] = np.nan

        # print(f"Prediction for {yesterday+timedelta(1)} -- ticker: {ticker} {'**resultat**'}\n {resultat.tail()}\n\n")

    Results = Predictions[['ticker', 'Date', 'y_test', 'Predicted for', 'y_pred', 'y_recommend', 'daily return %']]
    Results = Results.rename(
        columns={'ticker': 'Ticker', 'y_test': 'Observed', 'y_pred': 'Predicted', 'y_recommend': 'Recommended',
                 'daily return %': 'Daily return %'})

    return Results, Recommendations, avg_return


