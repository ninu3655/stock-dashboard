
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel

st.title("📈 Stock Price & Movement Predictor")

ticker = st.text_input("Enter stock ticker (e.g. AAPL, TSLA):", value="AAPL").upper()

if ticker:
    with st.spinner(f"Fetching data for {ticker}..."):
        # Download data
        df = yf.download(ticker, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
        if df.empty:
            st.error("No data found for this ticker. Try another.")
        else:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace=True)

            # Technical indicators
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
            df['Return'] = df['Close'].pct_change()

            # News sentiment
            analyzer = SentimentIntensityAnalyzer()
            news = yf.Ticker(ticker).news
            news_df = pd.DataFrame(news)
            if not news_df.empty:
                news_df['datetime'] = pd.to_datetime(news_df['providerPublishTime'], unit='s')
                news_df['date'] = news_df['datetime'].dt.date
                news_df['sentiment'] = news_df['title'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
                daily_sentiment = news_df.groupby('date')['sentiment'].mean()
            else:
                daily_sentiment = pd.Series(dtype=float)

            df['date'] = df.index.date
            df = df.merge(daily_sentiment, how='left', left_on='date', right_on='date')
            df['sentiment'].fillna(0, inplace=True)

            # Target for classification
            df['Target'] = df['Return'].shift(-1).apply(lambda x: 1 if x > 0.005 else (0 if x < -0.005 else 2))
            df.dropna(inplace=True)

            features = ['SMA_20', 'RSI', 'MACD', 'Return', 'sentiment']
            X = df[features]
            y = df['Target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

            clf = LGBMClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.markdown(f"### Movement Prediction Accuracy: {accuracy:.2%}")

            # Price forecasting with N-BEATS
            price_series = TimeSeries.from_dataframe(df, value_cols='Close')
            scaler = Scaler()
            price_series_scaled = scaler.fit_transform(price_series)
            train = price_series_scaled[:-100]
            model = NBEATSModel(input_chunk_length=30, output_chunk_length=7, n_epochs=50, random_state=42)
            model.fit(train)

            forecast = model.predict(n=7)
            forecast = scaler.inverse_transform(forecast)

            st.markdown("### Next 7 Days: Actual vs Predicted Closing Prices")
            result_df = pd.concat([df['Close'][-7:], forecast.pd_series()], axis=1)
            result_df.columns = ['Actual', 'Predicted']
            st.line_chart(result_df)

            st.dataframe(result_df)
