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
      menu_title="Maicln Menu",
      options=["Introduction", "Home"],
      icons=["book", "house"],
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
            scaler = MinMaxScaler(feature_range=(0,1))

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
                y_test.append(input_data[i,0])

            x_test, y_test = np.array(x_test), np.array(y_test)
            y_predicted = model.predict(x_test)
            scaler = scaler.scale_

            scale_factor = 1/scaler[0]
            y_predicted = y_predicted*scale_factor
            y_test = y_test*scale_factor

            # Final Graph
            st.subheader('Prediction vs Original')
            fig2 = plt.figure(figsize=(12,6))
            plt.plot(y_test, 'b', label='Original Price')
            plt.plot(y_predicted, 'r', label='Predicted Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig2)
        else:
            st.warning("Data for training is empty.")
    else:
        st.warning("Data not available. Please check the stock ticker or try again later.")

elif page == "Introduction":
    # Introduction page content
    st.title("Introduction to the Stock Trend Predictor")
    st.write("Welcome to our Website :wave:, your gateway to informed investment decisions powered by cutting-edge technology. We leverage the power of Long Short-Term Memory (LSTM) models, a form of Artificial Intelligence (AI), to analyze vast amounts of historical stock data and identify potential future trends.")
    st.write("This web app provides stock trend predictions based on historical data.")
    st.write("You can input a stock ticker symbol and view various visualizations and predictions.")
    st.subheader("Unlocking Market Insights")
    st.write("Our LSTM model goes beyond traditional technical indicators by capturing complex patterns within historical data. This allows us to provide you with insights that may not be readily apparent, potentially helping you stay ahead of the curve.")
    st.subheader("❗Disclaimer:")
    st.write("It's important to remember that stock market predictions are inherently probabilistic. Past performance is not necessarily indicative of future results. We encourage you to conduct thorough research and consider your own financial goals before making any investment decisions.")

