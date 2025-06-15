import streamlit as st
import pandas as pd
import io
from dotenv import load_dotenv
import sys
import os
from feature_utils import create_features, create_features_NotOpen, create_side_features, predict_today, remove_string_from_columns

# Load credentials from .env file
load_dotenv()
LOGIN_ID = os.getenv("LOGIN_ID")
LOGIN_PASS = os.getenv("LOGIN_PASS")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == LOGIN_ID and password == LOGIN_PASS:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

def run_main_and_capture_output():
    output = io.StringIO()
    sys.stdout = output  # stdoutを一時的にStringIOにリダイレクト

    # ====== あなたの main 関数の処理を関数として定義 ======
    tickers_urls = {
        "^N225": "https://www.investing.com/indices/japan-ni225"
    }

    tickers_urls_NotOpen = {
        "^GSPC": "https://finance.yahoo.com/quote/%5EGSPC/",       
        '^DJI': 'https://finance.yahoo.com/quote/%5EDJI/',
        '^IXIC': 'https://finance.yahoo.com/quote/%5EIXIC/',
        '^RUT':'https://finance.yahoo.com/quote/%5ERUT/',
        '^GDAXI': 'https://finance.yahoo.com/quote/%5EGDAXI/',
        '^HSI':'https://finance.yahoo.com/quote/%5EHSI/',
        '^KS11': 'https://finance.yahoo.com/quote/%5EKS11/'
    }

    tickers_urls_side = {
        "^VIX":"https://www.investing.com/indices/volatility-s-p-500",
        "^TNX":"https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield",
        "^TYX":"https://www.investing.com/rates-bonds/u.s.-30-year-bond-yield"
    }

    feature_columns = ['Open', 'Close_prev', 'Open_Close_diff_ratio', 'Close_Close_diff_ratio', 'Close_Open_diff_ratio', 'RSI', 'MACD Histogram','Open_diff_ratio_ByWeek','Open_diff_ratio_ByMonth']
    feature_columns_NotOpen = ['Open_Close_diff_ratio', 'Close_Close_diff_ratio', 'Close_Open_diff_ratio', 'RSI', 'MACD Histogram','Open_diff_ratio_ByWeek','Open_diff_ratio_ByMonth']
    features_columns_side = ['Open_prev','Open_Close_diff_ratio','Close_Close_diff_ratio']

    features = create_features(tickers_urls, feature_columns)
    features_NotOpen = create_features_NotOpen(tickers_urls_NotOpen, feature_columns_NotOpen)
    features_side = create_side_features(tickers_urls_side, features_columns_side)

    features.index = pd.to_datetime(features.index) 
    features_sorted = features.sort_index()

    features_NotOpen.index = pd.to_datetime(features_NotOpen.index) 
    features_NotOpen_sorted = features_NotOpen.sort_index()

    features_side.index = pd.to_datetime(features_side.index) 
    features_side_sorted = features_side.sort_index()

    merged_features = pd.concat([features_sorted, features_NotOpen_sorted, features_side_sorted], axis=1)
    merged_features = remove_string_from_columns(merged_features,'^N225_')
    merged_features.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
    merged_features.fillna(method='ffill', inplace=True)
    merged_features.columns = [col.lstrip('^') for col in merged_features.columns]
    merged_features_all = merged_features
    merged_features = merged_features[['Open_Close_diff_ratio', 'Close_Open_diff_ratio',
       'Open_diff_ratio_ByWeek', 'GSPC_Close_Close_diff_ratio',
       'GSPC_Open_diff_ratio_ByWeek', 'GSPC_Open_diff_ratio_ByMonth',
       'DJI_Close_Open_diff_ratio', 'DJI_Open_diff_ratio_ByWeek',
       'DJI_Open_diff_ratio_ByMonth', 'RUT_Open_diff_ratio_ByWeek',
       'GDAXI_Close_Open_diff_ratio', 'GDAXI_Open_diff_ratio_ByMonth',
       'KS11_Close_Open_diff_ratio', 'KS11_RSI', 'VIX_Open_prev',
       'VIX_Open_Close_diff_ratio', 'TNX_Open_prev',
       'TNX_Close_Close_diff_ratio', 'TYX_Open_prev']]

   
    # カラム名の先頭にある ^ を削除
    merged_features.columns = merged_features.columns.str.replace(r'^\^', '', regex=True)
    merged_features = merged_features.tail(30)

    pred = predict_today(merged_features)
    merged_features_all = pd.concat([merged_features_all,pred],axis=1)
    merged_features_all = merged_features_all.tail(10)

    merged_features_toInsert = merged_features_all.reset_index()
    merged_features_toInsert = merged_features_toInsert.rename(columns={'index': 'Date'})
    merged_features_toInsert['Date'] = merged_features_toInsert['Date'].dt.strftime('%Y-%m-%d')
    merged_features_toInsert = merged_features_toInsert.tail(10)
    merged_features_toInsert = merged_features_toInsert.where(merged_features_toInsert.notna(), None)
    merged_features_toInsert = merged_features_toInsert.replace({float('nan'): None, pd.NA: None, pd.NaT: None})
    
    if merged_features_toInsert.isna().any().any():
        print("DataFrameにまだNaN値があります：")
        print(merged_features_toInsert.isna().sum())
        merged_features_toInsert = merged_features_toInsert.dropna()

    for column in merged_features_toInsert.select_dtypes(include=['datetime', 'datetimetz']).columns:
        merged_features_toInsert[column] = merged_features_toInsert[column].dt.strftime('%Y-%m-%d %H:%M:%S')

    sys.stdout = sys.__stdout__  # stdoutを元に戻す
    return output.getvalue()


# ===== Streamlit UI部分 =====
def main_app():
    st.title("NI225 prediction AI")

    # Add a text input box for manual open price
    manual_open_price = st.text_input("Enter manual open price for ^N225 (leave blank to use default):", "")

    if st.button("Press here to predict daily move of NI225"):
        st.write("proccessing...")

        # Convert manual_open_price to float if provided, otherwise set to None
        manual_open_price = float(manual_open_price) if manual_open_price.strip() else None

        output = run_main_and_capture_output()
        st.text_area("Output", output, height=500)

# Auth gate
if not st.session_state.logged_in:
    login()
else:
    main_app()
