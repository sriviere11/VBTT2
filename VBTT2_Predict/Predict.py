# functions to predict model and calculate accuracy

import os
# import bamboolib as bam
import os.path

import joblib
import numpy as np
import pandas as pd

from VBTT2_Features.Features import preprocessing, initialize_data, create_predict_set
from VBTT2_SP500.SP500 import get_ticker_sector, generate_enhanced_data,YF_datetime
from VBTT2_IO.IO import read_config_file,delete_then_get_model_from_bucket,instantiate_logging
from VBTT2_Model.Model import Model_Train_Save, balanced_accuracy, model_predict, add_buy_sell_to_prediction



def predict_ticker(ticker_list_for_models):
    logger=instantiate_logging()
    ticker_list_for_models = ticker_list_for_models.split('-')
    # note when we use flask we will use /ticker/aapl-nflx-cdw
    # so we need to split



    # initialisation model
    # how many years for the model
    years = 6
    lags = 30  # how many days of lags we need of this model, this is like an hyperparameter for us
    version, additional_data, regressor, model, bucket,filename_json = read_config_file()  # other data needed


    nb_predict_days = 15  # size of test data in number of days

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
        logger.log_text( f"Function predict_ticker| Training model for at least one ticker in {ticker_list_for_models}")
        Model_Train_Save(ticker_list_for_models, years, lags, additional_data, nb_predict_days)
    else:
        logger.log_text(f"Function predict_ticker| no model Training is needed for  {ticker_list_for_models}")
        ## get variable for start the prediction
        ##just in case, we fetch 2 time the lags so that we don't have issue when lagging

    yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last, days = initialize_data(
        lags * 2.5, lags, nb_predict_days)

    ### validation if we have everything
    logger.log_text(f"Function predict_ticker|  Input ->ticker list {ticker_list_for_models}")
    logger.log_text(f"Function predict_ticker|  Input ->years {years}")
    logger.log_text(f"Function predict_ticker|  Input->lags {lags}")
    logger.log_text(f"Function predict_ticker|  Input ->predict days {nb_predict_days}")
    logger.log_text(f"Function predict_ticker|  Period->yesterday {yesterday}")
    logger.log_text(f"Function predict_ticker|  Period->train_date_start {train_date_start}")
    logger.log_text(f"Function predict_ticker|  Period->train_date_last {train_date_last}")
    logger.log_text(f"Function predict_ticker|  Period->test_date_start {test_date_start}")
    logger.log_text(f"Function predict_ticker|  Period->test_date_last {test_date_last}")


    logger.log_text(f"Function predict_ticker| Tickers for our model {ticker_list_for_models}")
    logger.log_text(f"Function predict_ticker| Additional data  we add to the tickers for the model {additional_data}")

    # df_tomorrow = preprocessing(ticker_list_for_models, additional_data, lags * 2)  # add additional features
    # df_tomorrow.shape

    # to store predictions
    Predictions = pd.DataFrame()

    filename_json=read_config_file()[5]
    for ticker in ticker_list_for_models:
        # load the saved model for the ticker
        filename = ticker + "_model.sav"
        loaded_model = joblib.load(filename)
        sector = get_ticker_sector(ticker,filename_json)
        additional_data = read_config_file()[1]
        additional_data = generate_enhanced_data(sector, ticker)
        df_tomorrow = preprocessing(ticker, additional_data, lags * 2.5)  # add additional features


        X_test, y_test, df_filtered = create_predict_set(ticker, df_tomorrow, lags, nb_predict_days,
                                                         additional_data)  # this is X and y



        temp_pred = model_predict(loaded_model, ticker, X_test, y_test)

        # ===
        temp_pred.drop(labels=[0], inplace=True)
        #temp_pred.to_csv('temp_pred.csv')

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



    # adding binary buy or sell to predictions dataframe
    Predictions = add_buy_sell_to_prediction(Predictions, ticker_list_for_models)


    DF_Recommendations = []
    blc_accuracy=[0]  # render for view html need a first element to be 0
    for ticker in ticker_list_for_models:
        blc_accuracy.append(round(balanced_accuracy(ticker, Predictions.iloc[:-1]),3))
        DF_Recommendations.append([ticker, "", "", "", balanced_accuracy(ticker, Predictions.iloc[:-1]),""])
        Recommendations = pd.DataFrame(DF_Recommendations,
                                       columns=["Ticker", 'Predicted for', 'Predicted', "Recommended", "Accuracy","Return%"])



    avg_return = [0]  # render for view html need a first element to be 0
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

        ticker_return = round(Predictions['daily return %'][Predictions['ticker'] == ticker].iloc[:-1].mean(), 2)
        Recommendations.loc[Recommendations['Ticker'] == ticker, 'Return%'] = ticker_return

        avg_return.append(ticker_return) # we are creating a list of the return
        # print(f"Prediction for {yesterday+timedelta(1)} -- ticker: {ticker} {'**resultat**'}\n {resultat.tail()}\n\n")


    Results = Predictions[['ticker', 'Date', 'y_test', 'Predicted for', 'y_pred', 'y_recommend', 'daily return %']]
    Results = Results.rename(
        columns={'ticker': 'Ticker', 'y_test': 'Observed', 'y_pred': 'Predicted', 'y_recommend': 'Recommended',
                 'daily return %': 'Return %'})

    date_range = [0]
    date_range.append(Results.iloc[0]['Date'].strftime("%Y/%m/%d"))
    date_range.append(Results.iloc[-2]['Date'].strftime("%Y/%m/%d"))

    return Results, Recommendations, avg_return,blc_accuracy,date_range



def generate_results(ticker,sector,Predictions):

    MODEL=read_config_file()[5]
    DF_Recommendations = []
    #blc_accuracy=[0]  # render for view html need a first element to be 0
    #blc_accuracy.append(round(balanced_accuracy(ticker, Predictions.iloc[:-1]),3))
    DF_Recommendations.append([ticker, "", "", "", balanced_accuracy(ticker, Predictions.iloc[:-1]),""])
    Recommendations = pd.DataFrame(DF_Recommendations, columns=["Ticker", 'Predicted for', 'Predicted', "Recommended", "Accuracy","Return%"])



    #avg_return = [0]  # render for view html need a first element to be 0
    # results = Predictions[['y_test', 'y_pred', 'ticker', 'y_predb','y_recommend']][Predictions['ticker'] == ticker]
    results = Predictions[['y_test', 'y_pred', 'ticker', 'y_predb', 'y_recommend', 'daily return %']][Predictions['ticker'] == ticker]
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

    ticker_return = round(Predictions['daily return %'][Predictions['ticker'] == ticker].iloc[:-1].mean(), 2)
    Recommendations.loc[Recommendations['Ticker'] == ticker, 'Return%'] = ticker_return

    #avg_return.append(ticker_return) # we are creating a list of the return
    # print(f"Prediction for {yesterday+timedelta(1)} -- ticker: {ticker} {'**resultat**'}\n {resultat.tail()}\n\n")


    Results = Predictions[['ticker', 'Date', 'y_test', 'Predicted for', 'y_pred', 'y_recommend', 'daily return %']]
    Results = Results.rename(
        columns={'ticker': 'Ticker', 'y_test': 'Observed', 'y_pred': 'Predicted', 'y_recommend': 'Recommended',
                 'daily return %': 'Return %'})

    Results['Sector']=sector
    Results['Model']=MODEL

    #below is not uses for this module but might be useful
    #date_range = [0]
    #date_range.append(Results.iloc[0]['Date'].strftime("%Y/%m/%d"))
    #date_range.append(Results.iloc[-2]['Date'].strftime("%Y/%m/%d"))

    return Results




