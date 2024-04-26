import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from keras.models import load_model
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Stock Trend Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Background style
st.markdown(
    """
    <style>
    .stApp {
        background: rgb(238,174,202);
        background: radial-gradient(circle, rgba(238,174,202,1) 0%, rgba(148,187,233,1) 100%);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Navbar
with st.sidebar:
    page = option_menu(
        menu_title="Main Menu",
        options=["Introduction", "Home", "Forecasting"],
        icons=["book", "house", "book"],
        # orientation="horizontal"
    )

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: none;
        color: #fff;
        text-align: center;
        padding: 10px 0;
    }
    </style>
    <div class="footer">
        <p>&copy; Stock Trend Prediction App</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Page content
if page == "Home":
    # Header
    st.title('Stock Trend Prediction')

    # Get user input
    user_input = st.text_input('Enter Stock Ticker', 'AAPL')

    # Fetch data
    yf.pdr_override()
    start_date = datetime(2014, 1, 1)
    end_date = datetime(2024, 3, 1)
    y_symbols = [user_input]
    df = pdr.get_data_yahoo(y_symbols, start=start_date, end=end_date)

    # Describing Data
    st.subheader('Data Summary')
    st.write(df.describe())

    # Visualizations
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df.Close)
    st.pyplot(fig)

    # Splitting Data into Training and Testing
    if not df.empty:
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

        if not data_training.empty:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))

            data_training_array = scaler.fit_transform(data_training)
            # Load my model
            model = load_model('keras_model.h5')

            # Testing part
            past_100_days = data_training.tail(100)
            final_df = past_100_days.append(data_testing, ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)
            y_predicted = model.predict(x_test)
            scaler = scaler.scale_

            scale_factor = 1/scaler[0]
            y_predicted = y_predicted*scale_factor
            y_test = y_test*scale_factor

            # Final Graph
            st.subheader('Prediction vs Original')
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(y_test, 'b', label='Original Price')
            plt.plot(y_predicted, 'r', label='Predicted Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig2)
        else:
            st.warning("Data for training is empty.")
    else:
        st.warning(
            "Data not available. Please check the stock ticker or try again later.")

elif page == "Forecasting":
    # Get user input
    user_input = st.text_input('Enter Stock Ticker', 'AAPL')

    key = "b36954810b5656782ecdbccad32bfdb9bd2e5b82"
    st.title('Stock Trend Forecasting')
    # user_input = st.text_input('Enter Stock Ticker', 'AAPL')
    df = pdr.get_data_tiingo(user_input, api_key=key)
    # df.to_csv('AAPL.csv')
    # df=pd.read_csv('AAPL.csv')
    df1 = df.reset_index()['close']
    st.subheader('Stock Price')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df1)
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    st.pyplot(fig)

    # LSTM are sensitive to the scale of the data. so we apply MinMax scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    # splitting dataset into train and test split
    training_size = int(len(df1)*0.65)
    test_size = len(df1)-training_size
    train_data, test_data = df1[0:training_size,
                                :], df1[training_size:len(df1), :1]
    # training_size,test_size
    # train_data
    # convert an array of values into a dataset matrix

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]  # i=0, 0,1,2,3-----99   100
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = load_model('keras_forecasting_model.h5')

    # Lets Do the prediction and check performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    # Transformback to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    # Calculate RMSE performance metrics
    import math
    from sklearn.metrics import mean_squared_error
    math.sqrt(mean_squared_error(y_train, train_predict))
    # Test Data RMSE
    math.sqrt(mean_squared_error(ytest, test_predict))
    # Plotting
    # shift train predictions for plotting
    look_back = 100
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2) +
                    1:len(df1)-1, :] = test_predict
    
    st.subheader('Actual vs Predicted Prices')
    # plot baseline and predictions
    fig = plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(df1), label='Actual Stock Prices')
    plt.plot(trainPredictPlot, label='Training Predicted Prices')
    plt.plot(testPredictPlot, label='Testing Predicted Prices')
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    st.pyplot(fig)

    x_input=test_data[341:].reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    # temp_input

    # demonstrate prediction for next 10 days
    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            # print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            # print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)

    st.subheader('Actual vs Forecasted Prices (Next 10 Days)')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    st.pyplot(fig)

    df3=df1.tolist()
    df3.extend(lst_output)
    st.subheader('Actual Stock Prices to comparison')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df3[1200:])
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    st.pyplot(fig)
    df3=scaler.inverse_transform(df3).tolist()
    st.subheader('Actual Stock Prices')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df3)
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    st.pyplot(fig)
    


elif page == "Introduction":
    # Introduction page content
    st.title("Introduction to the Stock Trend Predictor")
    st.write("Welcome to our Website :wave:, your gateway to informed investment decisions powered by cutting-edge technology. We leverage the power of Long Short-Term Memory (LSTM) models, a form of Artificial Intelligence (AI), to analyze vast amounts of historical stock data and identify potential future trends.")
    st.write("This web app provides stock trend predictions based on historical data.")
    st.write(
        "You can input a stock ticker symbol and view various visualizations and predictions.")
    st.subheader("Unlocking Market Insights")
    st.write("Our LSTM model goes beyond traditional technical indicators by capturing complex patterns within historical data. This allows us to provide you with insights that may not be readily apparent, potentially helping you stay ahead of the curve.")
    st.subheader("❗Disclaimer:")
    st.write("It's important to remember that stock market predictions are inherently probabilistic. Past performance is not necessarily indicative of future results. We encourage you to conduct thorough research and consider your own financial goals before making any investment decisions.")
