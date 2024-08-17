import streamlit as st
import yfinance as yf
import pytz
import pandas as pd
from datetime import datetime
from plotly import graph_objs as go
import numpy as np
np.float_ = np.float64
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="EG Stock Price",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)
# st.set_page_config(layout="wide",theme="light")
st.title('EG Stock Price')

@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.Ticker(ticker)
    df = stock_data.history(start=start_date, end=end_date)
    df = df.reset_index() 
    df = df.sort_index(ascending=False)
    return df

def plot_raw_data(df):
    if 'Date' in df.columns:  # Check if Date column exists
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],name='stock_close',line=dict(color='red') 
        ))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    else:
        st.error("The 'Date' column is missing from the dataset.")

def calculate_metrics(df):
    last_close = df['Close'].iloc[0]
    prev_close = df['Close'].iloc[1]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100 if prev_close != 0 else None
    high = df['High'].iloc[0]
    low = df['Low'].iloc[0]
    volume = df['Volume'].iloc[0]

    historical_high = df['High'].max()
    historical_low = df['Low'].min()
    historical_high_date = df[df['High'] == historical_high]['Date'].iloc[0]
    historical_low_date = df[df['Low'] == historical_low]['Date'].iloc[0]

    return last_close, change, pct_change, high, low, volume,historical_high, historical_low,historical_high_date,historical_low_date

auto_refresh_interval = 10 * 60  # 10 minutes in seconds
st.markdown(
    f"""
    <script>
    setTimeout(function() {{
        window.location.reload();
    }}, {auto_refresh_interval * 1000});
    </script>
    """,
    unsafe_allow_html=True
)
# malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
ticker = '8907.KL'
start_date = st.sidebar.date_input("Start Date", datetime(2001, 1, 1))  # (year, month, day)
end_date = datetime.today().strftime('%Y-%m-%d')
n_years = st.sidebar.slider("Years of prediction:",1,4)
period = n_years * 365


df = get_stock_data(ticker, start_date, end_date)
last_close,change,pct_change,high,low,volume,historical_high, historical_low,historical_high_date,historical_low_date = calculate_metrics(df)

st.metric(label=f"{ticker} Last Price", value=f"{last_close:.2f} MYR", delta=f"{change:.2f} ({pct_change:.2f}%)")
col1,col2,col3 = st.columns(3)
col1.metric("High", f"{high:.2f} MYR")
col2.metric("Low", f"{low:.2f} MYR")
col3.metric("Volume", f"{volume:,}") 
st.sidebar.metric("Historical High:",f"{historical_high:.2f} MYR")
st.sidebar.metric("Historical Low:",f"{historical_low:.2f} MYR")

st.write(f'Historical  Raw data for EG Industries:')
st.write(f"Dataset Shape: {df.shape}")
st.dataframe(df)
plot_raw_data(df)


# Forecasting
df_train = df[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})
df_train['ds'] = pd.to_datetime(df_train['ds']).dt.tz_localize(None) 

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
fig1.update_layout(xaxis_title="Time",yaxis_title="Stock Price")

st.plotly_chart(fig1)

st.write('Forecast Component')
fig2= m.plot_components(forecast)
st.write(fig2)



df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
df_p = df_cv.loc[:, ['ds', 'yhat']]
df_a = df_cv.loc[:, ['ds', 'y']]
df_p = df_p.merge(df_a, on='ds', how='inner')
mae = mean_absolute_error(df_p['y'], df_p['yhat'])
mse = mean_squared_error(df_p['y'], df_p['yhat'])
rmse = (mse)**(1/2)
# Display accuracy metrics
st.subheader('Prediction Accuracy')
st.write(f'MAE: {mae:.2f}')
st.write(f'MSE: {mse:.2f}')
st.write(f'RMSE: {rmse:.2f}')