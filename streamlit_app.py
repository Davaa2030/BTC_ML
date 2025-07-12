import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator
import joblib
import mplfinance as mpf
import numpy as np

# Load model
try:
    model = joblib.load("model_rfc_btc.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {str(e)}")
    st.stop()

st.title("📊 Prediksi Harga Bitcoin")

# Input tanggal
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Pilih Tanggal Mulai", pd.to_datetime("2022-01-01"))
with col2:
    end_date = st.date_input("Pilih Tanggal Akhir", pd.to_datetime("today"))

if start_date >= end_date:
    st.error("Tanggal akhir harus lebih dari tanggal mulai.")
    st.stop()

# Download data
try:
    df = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
    if df.empty:
        st.error("Tidak ada data yang ditemukan untuk rentang tanggal ini.")
        st.stop()
except Exception as e:
    st.error(f"Gagal mengambil data: {str(e)}")
    st.stop()

# Bersihkan data
df.dropna(inplace=True)

# Hitung EMA dengan error handling
try:
    # Versi 1: Menggunakan EMAIndicator dengan konversi eksplisit
    df["ema_50"] = EMAIndicator(close=df["Close"], window=50).ema_indicator().values
    df["ema_100"] = EMAIndicator(close=df["Close"], window=100).ema_indicator().values
    
    # Atau Versi 2: Menggunakan pandas rolling langsung (alternatif)
    # df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    # df["ema_100"] = df["Close"].ewm(span=100, adjust=False).mean()
    
    df.dropna(inplace=True)
except Exception as e:
    st.error(f"Gagal menghitung indikator teknikal: {str(e)}")
    st.stop()

# Prediksi
try:
    fitur = df[["ema_50", "ema_100"]]
    prediksi = model.predict(fitur)
    df["Prediksi"] = prediksi
except Exception as e:
    st.error(f"Gagal membuat prediksi: {str(e)}")
    st.stop()

# Klasifikasi sinyal
def klasifikasi(row):
    try:
        if row["Prediksi"] > row["Close"] * 1.01:
            return "BUY"
        elif row["Prediksi"] < row["Close"] * 0.99:
            return "SELL"
        else:
            return "HOLD"
    except:
        return "ERROR"

df["Sinyal"] = df.apply(klasifikasi, axis=1)

# Visualisasi
st.subheader("📈 Grafik Bitcoin")
try:
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
except Exception as e:
    st.error(f"Gagal membuat grafik: {str(e)}")

# Tampilkan data
st.subheader("📊 Data dan Prediksi")
st.dataframe(df[["Close", "ema_50", "ema_100", "Prediksi", "Sinyal"]])

# Download button
try:
    csv = df.to_csv().encode('utf-8')
    st.download_button(
        "⬇️ Download hasil ke CSV", 
        data=csv, 
        file_name='prediksi_btc.csv', 
        mime='text/csv'
    )
except Exception as e:
    st.error(f"Gagal menyiapkan data untuk download: {str(e)}")
