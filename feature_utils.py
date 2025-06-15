import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pickle
from google.cloud import storage
import os
from sklearn.ensemble import RandomForestClassifier

def create_features_NotOpen(tickers_urls, feature_columns):
    all_features = []

    for ticker, url in tickers_urls.items():
        # 現在の日付を取得
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=80)).strftime('%Y-%m-%d')

        # データを取得
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        data.columns = data.columns.droplevel(1)

        # データが最新か確認 (今日の日付がインデックスに存在するか)
        if end_date not in data.index.strftime('%Y-%m-%d'):
            # データが古い場合、scrape関数を呼び出してデータを取得し、dataに結合
            new_data = scrape_features_NotOpen(url)  # scrape関数は自分で実装する

            # new_rowをdataの下に追加
            new_row = pd.DataFrame(new_data, columns=data.columns, index=new_data.index)
            merged_df = pd.concat([data, new_row], ignore_index=False)
        else:
            merged_df = data
            # インデックスを日付型に変換
            merged_df.index = pd.to_datetime(merged_df.index, errors='coerce')

        # 'Open' と 'Close' 列を数値型に変換
        merged_df['Open'] = pd.to_numeric(merged_df['Open'], errors='coerce')
        merged_df['Close'] = pd.to_numeric(merged_df['Close'], errors='coerce')

        # 特徴量の作成
        merged_df['Close_prev'] = merged_df['Close'].shift(1)
        merged_df['Open_Close_diff_ratio'] = 100 * (merged_df['Open'].shift(1) - merged_df['Close'].shift(2)) / merged_df['Close'].shift(2)
        merged_df['Close_Close_diff_ratio'] = 100 * (merged_df['Close'].shift(1) - merged_df['Close'].shift(2)) / merged_df['Close'].shift(2)
        merged_df['Close_Open_diff_ratio'] = 100 * (merged_df['Close'].shift(1) - merged_df['Open'].shift(1)) / merged_df['Open'].shift(1)
        merged_df['Open_diff_ratio_ByWeek'] = 100 * (merged_df['Close'].shift(1) - merged_df['Open'].shift(5)) / merged_df['Open'].shift(5)
        merged_df['Open_diff_ratio_ByMonth'] = 100 * (merged_df['Close'].shift(1) - merged_df['Open'].shift(20)) / merged_df['Open'].shift(20)
        merged_df['Volume_prev'] = merged_df['Volume'].shift(1)
        merged_df['Volume_diff_ratio'] = 100 * (merged_df['Volume'].shift(1) - merged_df['Volume'].shift(2)) / merged_df['Volume'].shift(2)
        
        # RSIとMACDを計算
        merged_df['RSI'] = calculate_rsi(merged_df['Open'])
        merged_df['RSI'] = merged_df['RSI'].shift(1)
        macd = calculate_macd(merged_df['Open'])
        macd = macd.shift(1)
        merged_df = pd.concat([merged_df, macd], axis=1)

        # カラム名にティッカー名を追加
        merged_df = merged_df.add_prefix(f"{ticker}_")
        
        # 必要なカラムだけを抽出
        selected_columns = [f"{ticker}_{col}" for col in feature_columns]
        features_df = merged_df[selected_columns]
        
        all_features.append(features_df)

    # 全てのティッカーのデータをマージ
    final_df = pd.concat(all_features, axis=1)
    final_df.fillna(method='ffill', inplace=True)

    return final_df

def scrape_features_NotOpen(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    open_element = soup.find(class_='last-md last-lg yf-mrt107')
    live_element = soup.find('fin-streamer', class_='livePrice yf-1tejb6')

    if open_element:
        open_price = open_element.find('fin-streamer').get_text().strip().replace(',', '')
    else:
        open_price = None
        
    if live_element:
        live_price = live_element['data-value']
    else:
        live_price = None
    
    data = {
        'Open': [open_price],
        'Close':[live_price]
    }

    # Set the index to today's date
    today = datetime.today().strftime('%Y-%m-%d')
    df = pd.DataFrame(data, index=[today])

    return df 

def create_side_features(tickers_urls, feature_columns):
    all_features = []

    for ticker, url in tickers_urls.items():
        # 現在の日付を取得
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=80)).strftime('%Y-%m-%d')

        # データを取得
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        data.columns = data.columns.droplevel(1)

        # データが最新か確認 (今日の日付がインデックスに存在するか)
        if end_date not in data.index.strftime('%Y-%m-%d'):
            # データが古い場合、scrape関数を呼び出してデータを取得し、dataに結合
            new_data = scrape_side_features(url)  # scrape関数は自分で実装する

            # new_rowをdataの下に追加
            new_row = pd.DataFrame(new_data, columns=data.columns, index=new_data.index)
            merged_df = pd.concat([data, new_row], ignore_index=False)
        else:
            merged_df = data
            # インデックスを日付型に変換
            merged_df.index = pd.to_datetime(merged_df.index, errors='coerce')

        # 'Open' と 'Close' 列を数値型に変換
        merged_df['Open'] = pd.to_numeric(merged_df['Open'], errors='coerce')
        merged_df['Close'] = pd.to_numeric(merged_df['Close'], errors='coerce')

        # 特徴量の作成
        merged_df['Open_prev'] = merged_df['Open'].shift(1)
        merged_df['Open_Close_diff_ratio'] = 100 * (merged_df['Open'].shift(1) - merged_df['Close'].shift(2)) / merged_df['Close'].shift(2)
        merged_df['Close_Close_diff_ratio'] = 100 * (merged_df['Close'].shift(1) - merged_df['Close'].shift(2)) / merged_df['Close'].shift(2)
        merged_df['Close_Open_diff_ratio'] = 100 * (merged_df['Close'].shift(1) - merged_df['Open'].shift(1)) / merged_df['Open'].shift(1)

        # カラム名にティッカー名を追加
        merged_df = merged_df.add_prefix(f"{ticker}_")
        
        # 必要なカラムだけを抽出
        selected_columns = [f"{ticker}_{col}" for col in feature_columns]
        features_df = merged_df[selected_columns]
        
        all_features.append(features_df)

    # 全てのティッカーのデータをマージ
    final_df = pd.concat(all_features, axis=1)
    final_df.fillna(method='ffill', inplace=True)

    return final_df

def scrape_side_features(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 'data-test'が'open'の<dd>タグを検索
    open_element = soup.find(class_='mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 md:mb-0.5 md:gap-6')

    if open_element:
        # 必要な数値を含む2番目の<span>タグを取得
        value = open_element.find('div', {'data-test': 'instrument-price-last'})
        open_price = float(value.text.strip())
    else:
        print('None')

    data = {
        'Open': [open_price]
    }

    # Set the index to today's date
    today = datetime.today().strftime('%Y-%m-%d')
    df = pd.DataFrame(data, index=[today])

    return df

def create_features(tickers_urls, feature_columns, manual_open_price=None):
    all_features = []

    for ticker, url in tickers_urls.items():
        # 現在の日付を取得
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=80)).strftime('%Y-%m-%d')

        # データを取得
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        data.columns = data.columns.droplevel(1)

        # データが最新か確認 (今日の日付がインデックスに存在するか)
        if end_date not in data.index.strftime('%Y-%m-%d'):
            # データが古い場合、scrape関数を呼び出してデータを取得し、dataに結合
            new_data = scrape_features(url)  # scrape関数は自分で実装する

            # new_rowをdataの下に追加
            new_row = pd.DataFrame(new_data, columns=data.columns, index=new_data.index)
            merged_df = pd.concat([data, new_row], ignore_index=False)
        else:
            merged_df = data
            # インデックスを日付型に変換
            merged_df.index = pd.to_datetime(merged_df.index, errors='coerce')

        # 'Open' と 'Close' 列を数値型に変換
        merged_df['Open'] = pd.to_numeric(merged_df['Open'], errors='coerce')
        merged_df['Close'] = pd.to_numeric(merged_df['Close'], errors='coerce')

        # Use manual_open_price if provided
        if manual_open_price is not None:
            merged_df.loc[end_date, 'Open'] = manual_open_price

        # 特徴量の作成
        merged_df['Close_prev'] = merged_df['Close'].shift(1)
        merged_df['Open_Close_diff_ratio'] = 100 * (merged_df['Open'] - merged_df['Close'].shift(1)) / merged_df['Close'].shift(1)
        merged_df['Close_Close_diff_ratio'] = 100 * (merged_df['Close'].shift(1) - merged_df['Close'].shift(2)) / merged_df['Close'].shift(2)
        merged_df['Close_Open_diff_ratio'] = 100 * (merged_df['Close'].shift(1) - merged_df['Open'].shift(1)) / merged_df['Open'].shift(1)
        merged_df['Open_diff_ratio_ByWeek'] = 100 * (merged_df['Open'] - merged_df['Open'].shift(5)) / merged_df['Open'].shift(5)
        merged_df['Open_diff_ratio_ByMonth'] = 100 * (merged_df['Open'] - merged_df['Open'].shift(20)) / merged_df['Open'].shift(20)

        # RSIとMACDを計算
        merged_df['RSI'] = calculate_rsi(merged_df['Close'])
        merged_df['RSI'] = merged_df['RSI'].shift(1)
        macd = calculate_macd(merged_df['Close'])
        macd = macd.shift(1)
        merged_df = pd.concat([merged_df, macd], axis=1)

        # カラム名にティッカー名を追加
        merged_df = merged_df.add_prefix(f"{ticker}_")
        
        # 必要なカラムだけを抽出
        selected_columns = [f"{ticker}_{col}" for col in feature_columns]
        features_df = merged_df[selected_columns]
        
        all_features.append(features_df)

    # 全てのティッカーのデータをマージ
    final_df = pd.concat(all_features, axis=1)
    final_df.fillna(method='ffill', inplace=True)

    return final_df

def scrape_features(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    open_price = None  # デフォルト値として None を設定
    open_element = soup.find('dd', {'data-test': 'open'})

    if open_element:
        try:
            # 必要な値を含む 2 番目の <span> タグを取得
            value = open_element.find('span', class_='key-info_dd-numeric__ZQFIs').find_all('span')[1].text.strip()
            # カンマを除去して float に変換
            open_price = float(value.replace(',', ''))
        except (AttributeError, IndexError, ValueError) as e:
            print(f"オープン価格を取得中にエラーが発生しました: {e}")
    else:
        print('オープン価格の要素が見つかりませんでした。')

    if open_price is None:
        print("警告: オープン価格が取得できなかったため、NaN を設定します。")
        open_price = float('nan')  # 欠損値として NaN を設定

    data = {
        'Open': [open_price]
    }

    # インデックスを今日の日付に設定
    today = datetime.today().strftime('%Y-%m-%d')
    df = pd.DataFrame(data, index=[today])

    return df



def calculate_macd(open_prices, short_window=12, long_window=26, signal_window=9):
    # Calculate the short-term EMA using open prices
    short_ema = open_prices.ewm(span=short_window, adjust=False).mean()
    
    # Calculate the long-term EMA using open prices
    long_ema = open_prices.ewm(span=long_window, adjust=False).mean()
    
    # Calculate the MACD line
    macd_line = short_ema - long_ema
    
    # Calculate the signal line
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    
    # Calculate the MACD histogram
    macd_histogram = macd_line - signal_line

    # Create a DataFrame to hold the results
    result = pd.DataFrame({
        'MACD Line': macd_line,
        'Signal Line': signal_line,
        'MACD Histogram': macd_histogram
    })

    return result

def calculate_rsi(open_prices, window=14):
    # Calculate price changes
    delta = open_prices.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    # Calculate average gain and loss
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def remove_string_from_columns(df, string_to_remove):
    # カラム名をリストで取得し、指定された文字列を取り除く
    df.columns = [col.replace(string_to_remove, '') for col in df.columns]
    return df


def predict_today(X_input):
    """
    ロードしたモデルを使用して、指定された説明変数データで推論を行い、
    今日の日付の予測結果を返します。

    Parameters:
    X_input (DataFrame): 説明変数データのDataFrame

    Returns:
    DataFrame: 今日の日付のy_pred_original (予測結果)
    """
    # カレントディレクトリからモデルをロード
    with open('./best_rf_model_ni225_diff_maxitrade.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    with open('./NI225_RF_MetaLabel.pkl', 'rb') as f:
        loaded_model_meta = pickle.load(f)
    
    # ロードしたモデルで推論を実行
    y_pred_loaded = loaded_model.predict(X_input)

    # ロードしたモデルで推論を実行
    y_pred_loaded_Meta = loaded_model_meta.predict(X_input)

    # ラベルを元に戻す
    y_pred_original = y_pred_loaded - 1

    # ラベルを元に戻す
    y_pred_original_Meta = (y_pred_loaded_Meta + 1)*0.5

    y_pred_original = y_pred_original * y_pred_original_Meta

    # y_pred_originalをデータフレームに変換し、日付を追加
    y_pred_original_df = pd.DataFrame(y_pred_original, columns=['y_pred'], index=X_input.index)

    # y_predの最後のレコードを取得
    last_prediction = y_pred_original_df['y_pred'].iloc[-1]

    # 最後のレコードに基づいてアクションを決定してプリント
    if last_prediction == 1:
        print("Strong buy NI225")
    elif last_prediction == 0.5:
        print("Buy NI225")
    elif last_prediction == -0.5:
        print("Sell NI225")
    elif last_prediction == -1:
        print("Strong sell NI225")
    else:
        print("Do nothing for NI225")

    return y_pred_original_df