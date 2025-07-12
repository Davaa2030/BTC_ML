import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import joblib
import mplfinance as mpf
from ta.trend import EMAIndicator
import numpy as np

model = joblib.load("model_rfc_btc.pkl")

st.title("📊 Prediksi Harga Bitcoin")

start_date = st.date_input("Pilih Tanggal Mulai", pd.to_datetime("2022-01-01"))
end_date = st.date_input("Pilih Tanggal Akhir", pd.to_datetime("today"))

if start_date >= end_date:
    st.error("Tanggal akhir harus lebih dari tanggal mulai.")
else:
    df = yf.download("BTC-USD", start=start_date, end=end_date)
    df.dropna(inplace=True)

    ema_50 = EMAIndicator(close=df["Close"], window=50).ema_indicator()
    df["ema_50"] = ema_50.to_numpy().ravel()

    ema_100 = EMAIndicator(close=df["Close"], window=100).ema_indicator()
    df["ema_100"] = ema_100.to_numpy().ravel()
    df.dropna(inplace=True)
    
    fitur = df[["ema_50", "ema_100"]]
    prediksi = model.predict(fitur)
    df["Prediksi"] = prediksi

    def klasifikasi(row):
        if row["Prediksi"] > row["Close"] * 1.01:
            return "BUY"
        elif row["Prediksi"] < row["Close"] * 0.99:
            return "SELL"
        else:
            return "HOLD"

    df["Sinyal"] = df.apply(klasifikasi, axis=1)

    st.subheader("📈 Grafik Bitcoin")
    data_mpf = df[["Open", "High", "Low", "Close", "Volume"]]
    add_pred = mpf.make_addplot(df["Prediksi"], color='orange')

    fig, _ = mpf.plot(
        data_mpf,
        type='candle',
        style='charles',
        addplot=add_pred,
        volume=True,
        figsize=(10,6),
        returnfig=True
    )
    st.pyplot(fig)

    st.subheader("📊 Data dan Prediksi")
    st.dataframe(df[["Close", "ema_50", "ema_100", "Prediksi", "Sinyal"]])

    csv = df.to_csv().encode('utf-8')
    st.download_button("⬇️ Download hasil ke CSV", data=csv, file_name='prediksi_btc.csv', mime='text/csv')




