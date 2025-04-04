'''
Created Improved By Sadia Mazhar
13/03/2025
'''
from flask import Flask, render_template, request, flash, redirect, url_for

from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
import tweepy
import preprocessor as p
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import constants as ct
from Tweet import Tweet
import nltk
nltk.download('punkt')

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import os
os.environ['ALPHA_VANTAGE_API_KEY'] = 'N6A6QT6IBFJOPJ70'


import pandas as pd
import yfinance as yf
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries  # Import only if needed

# Replace with your actual Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "N6A6QT6IBFJOPJ70"

def get_historical(quote):
    end = datetime.now()
    start = datetime(end.year - 2, end.month, end.day)

    print(f"📡 Fetching data for {quote} from Yahoo Finance...")

    # Fetch data from Yahoo Finance
    df = yf.download(quote, start=start, end=end, auto_adjust=False)

    if not df.empty:
        df.reset_index(inplace=True)  # Ensure 'Date' is a column
        csv_filename = f"{quote}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"✅ Yahoo Finance Data Saved: {csv_filename}")

        # ✅ Just save the file locally instead
        print(f"✅ Data saved as: {csv_filename}")

        return df
    else:
        print("⚠ Yahoo Finance data not found! Trying Alpha Vantage...")

    # **Fallback: Try Alpha Vantage**
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, meta_data = ts.get_daily_adjusted(symbol=quote, outputsize='full')

        if data.empty:
            print(f"❌ Error: No data found for {quote} on Alpha Vantage either!")
            return None

        # Format Alpha Vantage Data
        data = data.head(503).iloc[::-1].reset_index()
        df = pd.DataFrame({
            "Date": data["date"],
            "Open": data["1. open"],
            "High": data["2. high"],
            "Low": data["3. low"],
            "Close": data["4. close"],
            "Adj Close": data["5. adjusted close"],
            "Volume": data["6. volume"]
        })

        # Save Data
        csv_filename = f"{quote}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"✅ Alpha Vantage Data Saved: {csv_filename}")

        # ✅ Just save the file locally instead
        print(f"✅ Data saved as: {csv_filename}")

        return df

    except Exception as e:
        print(f"❌ Error fetching data from Alpha Vantage: {e}")
        return None

# ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
from sklearn.metrics import mean_squared_error

# def ARIMA_ALGO(df):
#     uniqueVals = df["Code"].unique()
#     df["Date"] = pd.to_datetime(df["Date"])

#     results = {}

#     for company in uniqueVals[:10]:  # Run ARIMA for top 10 companies
#         data = df[df["Code"] == company].copy()
#         data['Price'] = data['Close'].fillna(method='ffill')  # Fill missing prices
#         Quantity_date = data[['Price', 'Date']].set_index("Date")

#         # Plot historical price trend
#         plt.figure(figsize=(7.2, 4.8), dpi=65)
#         plt.plot(Quantity_date, label=f"{company} Stock Trend")
#         plt.legend()
#         plt.savefig(f'Trends_{company}.png')  
#         plt.close()

#         # Train ARIMA on the entire dataset
#         quantity = Quantity_date["Price"].values
#         model = ARIMA(quantity, order=(6,1,0))  # Try tuning (p,d,q)
#         model_fit = model.fit()

#         # Predict next 7 days
#         future_steps = 7
#         future_forecast = model_fit.forecast(steps=future_steps)

#         # Generate future dates
#         last_date = Quantity_date.index[-1]
#         future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps+1)]

#         # Plot predictions
#         plt.figure(figsize=(7.2, 4.8), dpi=65)
#         plt.plot(Quantity_date.index, quantity, label="Actual Price", color="blue")
#         plt.plot(future_dates, future_forecast, label="Predicted Price (Next 7 Days)", color="red", linestyle="dashed")
#         plt.legend()
#         plt.savefig(f'ARIMA_{company}.png')  
#         plt.close()

#         # Store results
#         results[company] = {
#             "Predictions": future_forecast.tolist(),
#             "Last Known Price": quantity[-1],
#             "Predicted Next Price": future_forecast[0],
#         }

#         print(f"{company} - Last Price: {quantity[-1]}")
#         print(f"Predicted Next 7 Days: {future_forecast.tolist()}")
#         print("##############################################################################")

#     return results

 #************* LSTM SECTION **********************
# def LSTM_ALGO(df):
#         #Split data into training set and test set
#         dataset_train=df.iloc[0:int(0.8*len(df)),:]
#         dataset_test=df.iloc[int(0.8*len(df)):,:]
#         ############# NOTE #################
#         #TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
#         # HERE N=7
#         ###dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
#         training_set=df.iloc[:,4:5].values# 1:2, to store as numpy array else Series obj will be stored
#         #select cols using above manner to select as float64 type, view in var explorer

#         #Feature Scaling
#         from sklearn.preprocessing import MinMaxScaler
#         sc=MinMaxScaler(feature_range=(0,1))#Scaled values btween 0,1
#         training_set_scaled=sc.fit_transform(training_set)
#         #In scaling, fit_transform for training, transform for test

#         #Creating data stucture with 7 timesteps and 1 output.
#         #7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
#         X_train=[]#memory with 7 days from day i
#         y_train=[]#day i
#         for i in range(7,len(training_set_scaled)):
#             X_train.append(training_set_scaled[i-7:i,0])
#             y_train.append(training_set_scaled[i,0])
#         #Convert list to numpy arrays
#         X_train=np.array(X_train)
#         y_train=np.array(y_train)
#         X_forecast=np.array(X_train[-1,1:])
#         X_forecast=np.append(X_forecast,y_train[-1])
#         #Reshaping: Adding 3rd dimension
#         X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))#.shape 0=row,1=col
#         X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
#         #For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features)

#         #Building RNN
#         from keras.models import Sequential
#         from keras.layers import Dense
#         from keras.layers import Dropout
#         from keras.layers import LSTM

#         #Initialise RNN
#         regressor=Sequential()

#         #Add first LSTM layer
#         regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
#         #units=no. of neurons in layer
#         #input_shape=(timesteps,no. of cols/features)
#         #return_seq=True for sending recc memory. For last layer, retrun_seq=False since end of the line
#         regressor.add(Dropout(0.1))

#         #Add 2nd LSTM layer
#         regressor.add(LSTM(units=50,return_sequences=True))
#         regressor.add(Dropout(0.1))

#         #Add 3rd LSTM layer
#         regressor.add(LSTM(units=50,return_sequences=True))
#         regressor.add(Dropout(0.1))

#         #Add 4th LSTM layer
#         regressor.add(LSTM(units=50))
#         regressor.add(Dropout(0.1))

#         #Add o/p layer
#         regressor.add(Dense(units=1))

#         #Compile
#         regressor.compile(optimizer='adam',loss='mean_squared_error')

#         #Training
#         regressor.fit(X_train,y_train,epochs=25,batch_size=32 )
#         #For lstm, batch_size=power of 2

#         #Testing
#         ###dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
#         real_stock_price=dataset_test.iloc[:,4:5].values

#         #To predict, we need stock prices of 7 days before the test set
#         #So combine train and test set to get the entire data set
#         dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0)
#         testing_set=dataset_total[ len(dataset_total) -len(dataset_test) -7: ].values
#         testing_set=testing_set.reshape(-1,1)
#         #-1=till last row, (-1,1)=>(80,1). otherwise only (80,0)

#         #Feature scaling
#         testing_set=sc.transform(testing_set)

#         #Create data structure
#         X_test=[]
#         for i in range(7,len(testing_set)):
#             X_test.append(testing_set[i-7:i,0])
#             #Convert list to numpy arrays
#         X_test=np.array(X_test)

#         #Reshaping: Adding 3rd dimension
#         X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

#         #Testing Prediction
#         predicted_stock_price=regressor.predict(X_test)

#         #Getting original prices back from scaled values
#         predicted_stock_price=sc.inverse_transform(predicted_stock_price)
#         # fig = plt.figure(figsize=(7.2,4.8),dpi=65)
#         # plt.plot(real_stock_price,label='Actual Price')
#         # plt.plot(predicted_stock_price,label='Predicted Price')

#         # plt.legend(loc=4)
#         # plt.savefig('static/LSTM.png')
#         # plt.close(fig)
#         plt.figure(figsize=(7.2, 4.8), dpi=65)
#         plt.plot(dataset_test["Date"], real_stock_price, label="Actual Price")
#         plt.plot(dataset_test["Date"], predicted_stock_price, label="Predicted Price")
#         plt.xlabel("Date")
#         plt.ylabel("Stock Price")
#         plt.xticks(rotation=45)
#         plt.legend(loc=4)
#         plt.savefig('static/LSTM.png')
#         plt.close()


#         error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))


#         #Forecasting Prediction
#         forecasted_stock_price=regressor.predict(X_forecast)

#         #Getting original prices back from scaled values
#         forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)

#         lstm_pred=forecasted_stock_price[0,0]
#         print()
#         print("##############################################################################")
#         print("Tomorrow's ",quote," Closing Price Prediction by LSTM: ",lstm_pred)
#         print("LSTM RMSE:",error_lstm)
#         print("##############################################################################")
#         return lstm_pred,error_lstm
    
def LSTM_ALGO(df):
    dataset_train = df.iloc[0:int(0.8 * len(df)), :]
    dataset_test = df.iloc[int(0.8 * len(df)):, :]
    
    training_set = df.iloc[:, 4:5].values
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    
    X_train, y_train = [], []
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - 7:i, 0])
        y_train.append(training_set_scaled[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, epochs=25, batch_size=32)
    
    real_stock_price = dataset_test.iloc[:, 4:5].values
    dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
    testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
    testing_set = testing_set.reshape(-1, 1)
    testing_set = sc.transform(testing_set)
    
    X_test = []
    for i in range(7, len(testing_set)):
        X_test.append(testing_set[i - 7:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    
    # 7-day Forecasting
    forecast = []
    last_7_days = testing_set[-7:].reshape(1, 7, 1)
    for i in range(7):
        next_day_pred = regressor.predict(last_7_days)
        forecast.append(next_day_pred[0, 0])
        last_7_days = np.roll(last_7_days, -1)
        last_7_days[0, -1, 0] = next_day_pred
    
    forecasted_prices = sc.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    
    return predicted_stock_price, error_lstm, forecasted_prices
    
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd

# def ARIMA_ALGO(df):
#     try:
#         uniqueVals = df["Code"].unique()

#         # Ensure "Date" column is in correct format
#         if "Date" not in df.columns:
#             raise KeyError("Column 'Date' not found in DataFrame.")

#         df["Date"] = pd.to_datetime(df["Date"])

#         def arima_model(train, test):
#             history = [x for x in train]
#             predictions = []
#             for t in range(len(test)):
#                 model = ARIMA(history, order=(6,1,0))
#                 model_fit = model.fit()
#                 output = model_fit.forecast()
#                 predictions.append(output[0])
#                 history.append(test[t])
#             return predictions

#         for company in uniqueVals[:1]:  # Process only the first company
#             data = df[df["Code"] == company].copy()
#             data['Price'] = data['Close']
#             Quantity_date = data[['Price', 'Date']].set_index("Date")
#             Quantity_date = Quantity_date.fillna(method='ffill')

#             quantity = Quantity_date["Price"].values
#             size = int(len(quantity) * 0.80)
#             train, test = quantity[:size], quantity[size:]

#             if len(test) == 0 or len(train) == 0:
#                 print("⚠ Not enough data to train/test ARIMA model.")
#                 return None, None  # Ensure it returns 2 values

#             predictions = arima_model(train, test)

#             if len(predictions) < 2:
#                 print("⚠ Not enough predictions made by ARIMA.")
#                 return None, None  # Ensure it returns 2 values

#             error_arima = math.sqrt(mean_squared_error(test, predictions))
#             arima_pred = predictions[-2]

#             return arima_pred, error_arima  # Always return 2 values

#     except Exception as e:
#         print(f"❌ Error in ARIMA_ALGO: {str(e)}")
#         return None, None  # Ensure it returns 2 values even in case of failure

def ARIMA_ALGO(df):
    try:
        uniqueVals = df["Code"].unique()

        # Ensure "Date" column is in correct format
        if "Date" not in df.columns:
            raise KeyError("Column 'Date' not found in DataFrame.")

        df["Date"] = pd.to_datetime(df["Date"])

        def arima_model(train, test, days):
            history = [x for x in train]
            predictions = []

            # Predict for test set first (for RMSE calculation)
            for t in range(len(test)):
                model = ARIMA(history, order=(6,1,0))
                model_fit = model.fit()
                output = model_fit.forecast()
                predictions.append(output[0])
                history.append(test[t])

            # Forecast future days (multi-step forecast)
            future_forecast = []
            for _ in range(days):
                model = ARIMA(history, order=(6,1,0))
                model_fit = model.fit()
                output = model_fit.forecast()
                future_forecast.append(output[0])
                history.append(output[0])  # Append forecasted value to history

            return predictions, future_forecast

        for company in uniqueVals[:1]:  # Process only the first company
            data = df[df["Code"] == company].copy()
            data['Price'] = data['Close']
            Quantity_date = data[['Price', 'Date']].set_index("Date")
            Quantity_date = Quantity_date.fillna(method='ffill')

            quantity = Quantity_date["Price"].values
            size = int(len(quantity) * 0.80)
            train, test = quantity[:size], quantity[size:]

            if len(test) == 0 or len(train) == 0:
                print("⚠ Not enough data to train/test ARIMA model.")
                return None, None, None  # Ensure it returns 3 values

            # ✅ Define number of forecast days
            days = 7  # Set forecast days

            predictions, future_forecast = arima_model(train, test, days)

            if len(predictions) < 2:
                print("⚠ Not enough predictions made by ARIMA.")
                return None, None, None  # Ensure it returns 3 values

            error_arima = math.sqrt(mean_squared_error(test, predictions))
            arima_pred = predictions[-2]  # Tomorrow's prediction

            # ✅ Generate ARIMA Graph & Save
            min_len = min(len(test), len(predictions))
            plt.figure(figsize=(7.2, 4.8), dpi=65)
            plt.plot(df["Date"].values[-min_len:], test[:min_len], label="Actual Price", color="blue")
            plt.plot(df["Date"].values[-min_len:], predictions[:min_len], label="Predicted Price", color="red")
            plt.xlabel("Date")
            plt.ylabel("Stock Price")
            plt.xticks(rotation=45)
            plt.legend(loc=4)
            plt.title(f"ARIMA Model Predictions for {company}")
            plt.savefig("static/ARIMA.png")
            plt.close()

            return arima_pred, error_arima, future_forecast  # Returns tomorrow's prediction, error, and 7-day forecast

    except Exception as e:
        print(f"❌ Error in ARIMA_ALGO: {e}")
        return None, None, None  # Ensure it always returns 3 values
    
def LIN_REG_ALGO(df):
        #No of days to be forcasted in future
        forecast_out = int(7)
        #Price after n days
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        #New df with only relevant data
        df_new=df[['Close','Close after n days']]

        #Structure data for train, test & forecast
        #lables of known data, discard last 35 rows
        y =np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
        #all cols of known data except lables, discard last 35 rows
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
        #Unknown, X to be forecasted
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])

        #Traning, testing to plot graphs, check accuracy
        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]

        # Feature Scaling===Normalization
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        X_to_be_forecasted=sc.transform(X_to_be_forecasted)

        #Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)

        #Testing
        y_test_pred=clf.predict(X_test)
        y_test_pred=y_test_pred*(1.04)
        import matplotlib.pyplot as plt2
        # fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
        # plt2.plot(y_test,label='Actual Price' )
        # plt2.plot(y_test_pred,label='Predicted Price')

        # plt2.legend(loc=4)
        # plt2.savefig('static/LR.png')
        # plt2.close(fig)
        # Ensure X_test and y_test have the same length
        min_len = min(len(df.iloc[int(0.8 * len(df)):, :]["Date"]), len(y_test), len(y_test_pred))

        plt2.figure(figsize=(7.2, 4.8), dpi=65)
        plt2.plot(df.iloc[int(0.8 * len(df)):, :]["Date"].values[:min_len], y_test[:min_len], label="Actual Price")
        plt2.plot(df.iloc[int(0.8 * len(df)):, :]["Date"].values[:min_len], y_test_pred[:min_len], label="Predicted Price")
        plt2.xlabel("Date")
        plt2.ylabel("Stock Price")
        plt2.xticks(rotation=45)
        plt2.legend(loc=4)
        plt2.savefig('static/LR.png')
        plt2.close()


        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))


        #Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set=forecast_set*(1.04)
        mean=forecast_set.mean()
        lr_pred=forecast_set[0,0]
        print()
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by Linear Regression: ",lr_pred)
        print("Linear Regression RMSE:",error_lr)
        print("##############################################################################")
        return df, lr_pred, forecast_set, mean, error_lr
    
    #**************** SENTIMENT ANALYSIS **************************
def retrieving_tweets_polarity(symbol):
        stock_ticker_map = pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')
        stock_full_form = stock_ticker_map[stock_ticker_map['Ticker']==symbol]
        symbol = stock_full_form['Name'].to_list()[0][0:12]

        auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
        auth.set_access_token(ct.access_token, ct.access_token_secret)
        user = tweepy.API(auth)

        tweets = tweepy.Cursor(user.search_tweets, q=symbol, tweet_mode='extended', lang='en',exclude_replies=True).items(ct.num_of_tweets)

        tweet_list = [] #List of tweets alongside polarity
        global_polarity = 0 #Polarity of all tweets === Sum of polarities of individual tweets
        tw_list=[] #List of tweets only => to be displayed on web page
        #Count Positive, Negative to plot pie chart
        pos=0 #Num of pos tweets
        neg=1 #Num of negative tweets
        for tweet in tweets:
            count=20 #Num of tweets to be displayed on web page
            #Convert to Textblob format for assigning polarity
            tw2 = tweet.full_text
            tw = tweet.full_text
            #Clean
            tw=p.clean(tw)
            #print("-------------------------------CLEANED TWEET-----------------------------")
            #print(tw)
            #Replace &amp; by &
            tw=re.sub('&amp;','&',tw)
            #Remove :
            tw=re.sub(':','',tw)
            #print("-------------------------------TWEET AFTER REGEX MATCHING-----------------------------")
            #print(tw)
            #Remove Emojis and Hindi Characters
            tw=tw.encode('ascii', 'ignore').decode('ascii')

            #print("-------------------------------TWEET AFTER REMOVING NON ASCII CHARS-----------------------------")
            #print(tw)
            blob = TextBlob(tw)
            polarity = 0 #Polarity of single individual tweet
            for sentence in blob.sentences:

                polarity += sentence.sentiment.polarity
                if polarity>0:
                    pos=pos+1
                if polarity<0:
                    neg=neg+1

                global_polarity += sentence.sentiment.polarity
            if count > 0:
                tw_list.append(tw2)

            tweet_list.append(Tweet(tw, polarity))
            count=count-1
        if len(tweet_list) != 0:
            global_polarity = global_polarity / len(tweet_list)
        else:
            global_polarity = global_polarity
        neutral = ct.num_of_tweets - pos - neg
        if neutral < 0:
            neg = neg + neutral
            neutral = 20

        print()
        print("##############################################################################")

        print("Positive Tweets :",pos,"Negative Tweets :",neg,"Neutral Tweets :",neutral)
        print("##############################################################################")
        labels=['Positive','Negative','Neutral']
        sizes = [pos,neg,neutral]
        explode = (0, 0, 0)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        fig1, ax1 = plt.subplots(figsize=(7.2,4.8),dpi=65)
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        plt.tight_layout()
        plt.savefig('static/SA.png')
        plt.close(fig)
        #plt.show()
        if global_polarity>0:
            print()
            print("##############################################################################")
            print("Tweets Polarity: Overall Positive")
            print("##############################################################################")
            tw_pol="Overall Positive"
        else:
            print()
            print("##############################################################################")
            print("Tweets Polarity: Overall Negative")
            print("##############################################################################")
            tw_pol="Overall Negative"
        return global_polarity,tw_list,tw_pol,pos,neg,neutral
    
def recommending(df, global_polarity,today_stock,mean):
        if today_stock.iloc[-1]['Close'] < mean:
            if global_polarity > 0:
                idea="RISE"
                decision="BUY"
                print()
                print("##############################################################################")
                print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
            elif global_polarity <= 0:
                idea="FALL"
                decision="SELL"
                print()
                print("##############################################################################")
                print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
        else:
            idea="FALL"
            decision="SELL"
            print()
            print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
        return idea, decision
import os
import pandas as pd
import numpy as np

def get_stock_prediction(quote):
    """
    Fetches historical stock data, preprocesses it, and runs prediction models.

    Args:
        quote (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT').

    Returns:
        dict: Stock prediction results.
    """

    # ✅ Fetch stock data
    try:
        get_historical(quote)  # Ensure the stock data file is saved
    except Exception as e:
        return {"error": f"Failed to fetch stock data: {e}"}

    # ✅ Load CSV file
    try:
        df = pd.read_csv(f"{quote}.csv")
    except Exception as e:
        return {"error": f"Error reading CSV file for {quote}: {e}"}

    # ✅ Ensure required columns exist
    required_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if not all(col in df.columns for col in required_columns):
        return {"error": "Missing required columns in dataset."}

    # ✅ Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # ✅ Ensure numeric conversion for ARIMA
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ✅ Drop any remaining NaN values
    df = df.dropna()

    # ✅ Display today's stock data
    print("##############################################################################")
    print(f"Today's {quote} Stock Data:")
    today_stock = df.iloc[-1:]  # Get last row
    print(today_stock)
    print("##############################################################################")

    # ✅ Add stock ticker for reference
    df["Code"] = quote

    # ✅ Run models with error handling
    # try:
    #     arima_pred, error_arima = ARIMA_ALGO(df)
    # except Exception as e:
    #     arima_pred, error_arima = None, None
    #     print(f"❌ Error in ARIMA_ALGO: {e}")
    try:
        arima_pred, error_arima, arima_forecast = ARIMA_ALGO(df)
    except Exception as e:
     arima_pred, error_arima, arima_forecast = None, None, []
     print(f"❌ Error in ARIMA_ALGO: {e}")

    try:
        predicted_price, error_lstm, future_prices = LSTM_ALGO(df)
    except Exception as e:
        predicted_price, error_lstm, future_prices = None, None, []
        print(f"❌ Error in LSTM_ALGO: {e}")


    try:
        df, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df)
    except Exception as e:
        lr_pred, forecast_set, mean, error_lr = None, [], None, None
        print(f"❌ Error in LIN_REG_ALGO: {e}")

    # Twitter Lookup is no longer free
    polarity, tw_list, tw_pol, pos, neg, neutral = 0, [], "Can't fetch tweets, Twitter Lookup is no longer free in API v2.", 0, 0, 0

    # ✅ Ensure Matplotlib doesn't break due to categorical data
    try:
        idea, decision = recommending(df, polarity, today_stock, mean)
    except Exception as e:
        idea, decision = None, None
        print(f"❌ Error in recommending(): {e}")

    print("\nForecasted Prices for Next 7 days:")
    print(forecast_set)

    today_stock = today_stock.round(2)

    # ✅ Ensure 'Volume' column exists before using it
    volume_str = today_stock["Volume"].iloc[0] if "Volume" in today_stock.columns else "N/A"

    # ✅ Return stock predictions as a dictionary
    return {
        "quote": quote,
        "arima_pred": round(float(arima_pred), 2) if arima_pred else None,
        "lstm_pred": round(float(predicted_price[0]), 2) if isinstance(predicted_price, (list, np.ndarray)) and len(predicted_price) > 0 else None,
        "lstm_7days": future_prices.tolist() if isinstance(future_prices, np.ndarray) else (future_prices if isinstance(future_prices, list) else []),
        "lr_pred": round(float(lr_pred), 2) if lr_pred else None,
        "open": float(today_stock["Open"].iloc[0]),
        "close": float(today_stock["Close"].iloc[0]),
        "adj_close": float(today_stock["Adj Close"].iloc[0]),
        "high": float(today_stock["High"].iloc[0]),
        "low": float(today_stock["Low"].iloc[0]),
        "volume": volume_str,
        "forecast_set": forecast_set.tolist() if isinstance(forecast_set, np.ndarray) else (forecast_set if len(forecast_set) > 0 else []),
        # Add ARIMA's 7-day forecast in return statement
        "arima_forecast": arima_forecast if arima_forecast else [],
        "error_lr": round(float(error_lr), 2) if error_lr else None,
        "error_lstm": round(float(error_lstm), 2) if error_lstm else None,
        "error_arima": round(float(error_arima), 2) if error_arima else None,
        "idea": idea,
        "decision": decision
    }
import os

# Ensure 'static' directory exists
if not os.path.exists("static"):
    os.makedirs("static")  # Create the directory if it doesn't exist



# quote = "MSFT"
# result = get_stock_prediction(quote)
# print(result)
