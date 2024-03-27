import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
# from keras.models import load_model
import streamlit as st
from datetime import datetime

yf.pdr_override()
start_date = datetime(2010,1,1)
end_date = datetime(2019,12,31)

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
y_symbols = []
y_symbols.append(user_input)
df = pdr.get_data_yahoo(y_symbols, start=start_date, end=end_date)

# Describing Data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

