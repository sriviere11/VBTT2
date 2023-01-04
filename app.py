import os
import os.path
import numpy as np
from yahoo_fin import stock_info as si
from flask import *
from VBTT2_IO.IO import read_list,delete_then_get_model_from_bucket,delete_blob,read_config_file
from VBTT2_Predict.Predict import predict_ticker
from VBTT2_SP500.SP500 import get_SP500
###########################################
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return f'</br><b><FONT COLOR="#228335">Welcome to VBTT v.0 ===> Please  use a better route:!</FONT></b>   </b> Example: /ticker/your_ticker_here</b> ' \
f'</br><FONT COLOR="#228335">==================================================================</FONT>' \
f'</br></br><b><FONT COLOR="#BE5313"> Find ticker (Symbols) here --></FONT></b> <a href="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies">https://en.wikipedia.org/wiki/List_of_S%26P_500_companies</a> ' \
f'</br></br></br></br><b><FONT COLOR="#377ba8">Other examples of commands to use with VBTT:</FONT></b>' \
f'</br><b><FONT COLOR="#377ba8">=====================================</FONT></b>' \
f'</br><b>/ticker/aapl</b>           ---> will output recommendations for 1 ticker AAPL.' \
    f'</br></br><b>/ticker/aapl-cdw-goog</b> ---> will output recommendations for the 3 tickers AAPL, CDW, GOOG.' \
f'</br></br></br>  ' \
    f'</br><b>/details/aapl</b>           ---> will output details stats and prediction analysis for 1 ticker AAPL.' \
    f'</br></br><b>/details/aapl-cdw-goog</b>  ---> will output details stats and prediction analysis for the 3 tickers AAPL, CDW, GOOG.' \
f'</br></br></br>  ' \
    f'</br><b>/help</b>                   ---> will output help info similar to this page' \
f'</br></br></br>  ' \
f'</br> ' \
f'</br><b><FONT COLOR="#377ba8">Disclaimer</FONT></b>' \
f'</br><b><FONT COLOR="#377ba8">========</FONT></b>' \
f'</br>The author is not responsible for any loss that may occur in using the VBTT tool.' \
f'</br>This tool is the result of university assignment and as such you should not use' \
f'</br>this tool in real life or production environment.' \
f'</br></br></br>  ' \
f'</br></br><b><FONT COLOR="#228335">VBTT enjoy! </br> (Nov 2022)</FONT></b>'



@app.route('/help', methods=['GET'])
def help():
    return f'</br><b><FONT COLOR="#228335">Welcome to VBTT v.0 ===> Please  use a better route:!</FONT></b>   </b> Example: /ticker/your_ticker_here</b> ' \
f'</br><FONT COLOR="#228335">==================================================================</FONT>' \
f'</br></br><b><FONT COLOR="#BE5313"> Find ticker (Symbols) here --></FONT></b> <a href="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies">https://en.wikipedia.org/wiki/List_of_S%26P_500_companies</a> ' \
f'</br></br></br></br><b><FONT COLOR="#377ba8">Other examples of commands to use with VBTT:</FONT></b>' \
f'</br><b><FONT COLOR="#377ba8">=====================================</FONT></b>' \
f'</br><b>/ticker/aapl</b>           ---> will output recommendations for 1 ticker AAPL.' \
    f'</br></br><b>/ticker/aapl-cdw-goog</b> ---> will output recommendations for the 3 tickers AAPL, CDW, GOOG.' \
f'</br></br></br>  ' \
    f'</br><b>/details/aapl</b>           ---> will output details stats and prediction analysis for 1 ticker AAPL.' \
    f'</br></br><b>/details/aapl-cdw-goog</b>  ---> will output details stats and prediction analysis for the 3 tickers AAPL, CDW, GOOG.' \
f'</br></br></br>  ' \
    f'</br><b>/help</b>                   ---> will output help info similar to this page' \
f'</br></br></br>  ' \
f'</br> ' \
f'</br><b><FONT COLOR="#377ba8">Disclaimer</FONT></b>' \
f'</br><b><FONT COLOR="#377ba8">========</FONT></b>' \
f'</br>The author is not responsible for any loss that may occur in using the VBTT tool.' \
f'</br>This tool is the result of university assignment and as such you should not use' \
f'</br>this tool in real life or production environment.' \
f'</br></br></br>  ' \
f'</br></br><b><FONT COLOR="#228335">VBTT enjoy! </br> (Nov 2022)</FONT></b>'




@app.route('/get_sp500', methods=['GET'])
def get_sp500():
    SP500_tickers = si.tickers_sp500()
    filename_json=read_config_file()[5]
    delete_blob(filename_json)
    get_SP500(filename_json)
    return f"JSON was generated  successfully- try again with /ticker/xxx or /details/xxx (where xxx is your list of tickers separated by '-'."



@app.route('/ticker/<ticker>', methods=['GET'])
def ticker(ticker):
    ticker = ticker.upper()
    version=read_config_file()[0]
    model_html=read_config_file()[3]
    filename_json=read_config_file()[5]
    check=delete_then_get_model_from_bucket(filename_json)  # delete local file and copy file from bucket to have fresh one
    #file_exists = os.path.exists("SP500.json") # no longer required with blob check above
    if check==True:  #file json exist
        SP500_list = read_list(filename_json)
        SP500_list = np.array(SP500_list)  # we need an array
        validation=all([([x] in SP500_list[:,:1]) for x in ticker.split("-")])
        # all allow to check if a list o bolean is tru or false . all([true, false, true....etc])
	    # X in SP50_list, etc.... .... will check if x is in my SP500 list. here all row and column 0
	    # for x in is selecting each at a time
    
        if validation==True:
            Results, Recommendations,avg_return,blc_accuracy,date_range = predict_ticker(ticker)
            data = Recommendations
            data.set_index(['Ticker'], inplace=True)
            data.index.name = None
            return render_template('view_ticker.html', tables=[data.to_html(classes='Recommendations')],\
                                   titles=['na', ticker],version='Version '+version,model_html=model_html, avg_return=avg_return,blc_accuracy=blc_accuracy,date_range=date_range)

        else:
            return f"Incorrect ticker, please fix or select another."
    else:
        return f"JSON does not exists - Please generate JSON  with /get_SP500"


@app.route('/details/<ticker>', methods=['GET'])
def details(ticker):
    ticker = ticker.upper()
    filename_json = read_config_file()[5]
    version=read_config_file()[0]
    model_html=read_config_file()[3]
    check=delete_then_get_model_from_bucket(filename_json)  # delete local file and copy file from bucket to have fresh one
    #file_exists = os.path.exists("SP500.json")  # no longer required with blob check above
    if check==True:  #json file exists
        print('how come?')
        SP500_list = read_list(filename_json)
        SP500_list = np.array(SP500_list)  # we need an array
        validation = all([([x] in SP500_list[:, :1]) for x in ticker.split("-")])
        # all allow to check if a list o bolean is tru or false . all([true, false, true....etc])
        # X in SP50_list, etc.... .... will check if x is in my SP500 list. here all row and column 0
        # for x in is selecting each at a time

        if validation == True:
            Results, Recommendations,avg_return,blc_accuracy,date_range = predict_ticker(ticker)
            data = Results  # we call this data to be able to use it in html render code we took from internet

            # data.set_index(['Ticker'], inplace=True)
            data.index.name = None
            ticker = ticker.split('-')  # this will transform aapl-nflx-cdw to ['aapl','nflx','cdw']
            result_string = [Results[Results['Ticker'] == t].to_html(classes=t) for t in ticker]
            # we need to be able to create something that will be used on the function in comment below.
            return render_template('view_details.html', tables=result_string, titles=['na'] + ticker,\
                                   version='Version '+version,model_html=model_html,avg_return=avg_return,blc_accuracy=blc_accuracy,date_range=date_range)

        else:
            return f"Incorrect ticket, please fix or select another."
    else:
        return f"JSON does not exists - Please generate JSON  with /get_SP500"

    """"
    #return render_template('view_details.html', tables= [Results[Results['Ticker']=='NFLX'].to_html(classes='NFLX'),
                                                 Results[Results['Ticker']=='CDW'].to_html(classes='CDW')],
                                                titles=['na', 'CDW', 'AAPL'])

    
    return render_template('view_details.html',tables=[Results[Results['Ticker']=='CDW'].to_html(classes='CDW'),
                                                Results[Results['Ticker']=='AAPL'].to_html(classes='AAPL'),
                                                Results[Results['Ticker']=='NFLX'].to_html(classes='NFLX')],
                                                titles=['na','CDW','AAPL','NFLX'])
    """





if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='0.0.0.0', port=8080, debug=True)
