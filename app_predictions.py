import os
import os.path
import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si
from VBTT2_IO.IO import read_list,delete_then_get_model_from_bucket,delete_blob,read_config_file,instantiate_logging,save_dataframe_to_bucket,read_dataframe_from_bucket,upload_file_to_bucket
from VBTT2_Predict.Predict import predict_ticker,YF_datetime,generate_results
from VBTT2_SP500.SP500 import read_create_write_SP500, get_SP500,get_all_tickers_sector,get_all_tickers_industry
from VBTT2_Features.Features import get_yf_dataframe,initialize_data,create_train_test_set,create_predict_set
from VBTT2_Model.Model import  model_predict, add_buy_sell_to_prediction,balanced_accuracy,select_regressor



os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/steve/Downloads/creds.json"








logger =instantiate_logging()

predictions = pd.DataFrame()  # to store predictionsdfpredictions = pd.DataFrame()  # to store predictions
Results =pd.DataFrame()


version, additional_data, regressor, MODEL,bucket,filename_json=read_config_file()
MODEL=select_regressor(MODEL)

filename_json=read_config_file()[5]

SP500_list=read_list(filename_json)
SP500_list = np.array(SP500_list,dtype=object)
unique_SP500_sectors=np.unique(SP500_list[:,1])
unique_SP500_tickers=np.unique(SP500_list[:,0])

years=5
days=years*360
nb_predict_days=30
lags=30

yesterday, start_date, train_date_start, train_date_last, test_date_start, test_date_last,days=initialize_data(days,lags,nb_predict_days)


for sector in unique_SP500_sectors:

    ticker_list_for_models =get_all_tickers_sector(sector).tolist()


    additional_data = read_config_file()[1]
    tickers_in_sector_extended = np.concatenate((ticker_list_for_models, additional_data), axis=None)
    tickers_in_sector_extended = tickers_in_sector_extended.tolist()

    ### validation if we have everything
    logger.log_text(f"Module app_predictions|  Input ->Sector {sector}")
    logger.log_text(f"Module app_predictions|  Input ->ticker list {ticker_list_for_models}")
    logger.log_text(f"Module app_predictions|  Input ->ticker list extended {tickers_in_sector_extended}")
    logger.log_text(f"Module app_predictions|  Input ->years {years}")
    logger.log_text(f"Module app_predictions|  Input->lags {lags}")
    logger.log_text(f"Module app_predictions|  Input ->predict days {nb_predict_days}")
    logger.log_text(f"Module app_predictions|  Period->yesterday {yesterday}")
    logger.log_text(f"Module app_predictions|  Period->train_date_start {train_date_start}")
    logger.log_text(f"Module app_predictions|  Period->train_date_last {train_date_last}")
    logger.log_text(f"Module app_predictions|  Period->test_date_start {test_date_start}")
    logger.log_text(f"Module app_predictions|  Period->test_date_last {test_date_last}")

    logger.log_text(f"Module app_predictions| Tickers for our model {ticker_list_for_models}")
    logger.log_text \
        (f"Module app_predictions| Additional data  we add to the tickers for the model {additional_data}")

    logger.log_text \
        (f"Module app_predictions| Model VALIDATE - This is the number of training days of the train dataset {days}")







    matrix_features_sector = get_yf_dataframe(tickers_in_sector_extended, days)
    matrix_features_sector = matrix_features_sector.reindex(sorted(matrix_features_sector.columns), axis=1)


    for ticker in ticker_list_for_models:



        X_train, y_train, X_test, y_test, df_filtered = create_train_test_set(ticker, matrix_features_sector, lags, additional_data, days, nb_predict_days)
        print(MODEL)
        print(ticker)
        MODEL.fit(X_train, y_train)
        # save the model to disk
        # filename = ticker + '_modelnew.sav'
        # joblib.dump(MODEL, filename)
        # upload_file_to_bucket(filename)

        X_test, y_test, df_filtered = create_predict_set(ticker, matrix_features_sector, lags, nb_predict_days
                                                         ,additional_data)

        temp_pred = model_predict(MODEL, ticker, X_test, y_test)


        # ===
        temp_pred.drop(labels=[0], inplace=True)


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

        # adding other information
        temp_pred2['Sector' ] =sector
        temp_pred2["Model" ] =MODEL
        # ===
        predictions = predictions.append(temp_pred2, ignore_index=True)  # this is to store in the master pandas list


        # add binary buy=1 and sell=0
        Predictions = add_buy_sell_to_prediction(predictions ,[ticker])
        # lag_lagged is a DF containing predictions + buy and Sell label
        # print("Predictions -->",Predictions)




        Results_for_ticker =generate_results(ticker ,sector ,Predictions)
        Results = Results.append(Results_for_ticker.iloc[-1], ignore_index=True)  # this is to store in the master pandas list. Not [-1] because we want to add only the latest prediction
        logger.log_text(f"Module app_predictions|  Generated  Predictions {Predictions.shape} Results {Results.shape}-> {ticker} in {sector}")
        logger.log_text(f"Module app_predictions|  Results -> {Results_for_ticker.iloc[-1]}")