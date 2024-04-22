from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl import config_tickers
from finrl.config import INDICATORS
import itertools


def download_stock_data(train_start, train_end, trade_start, trade_end):
    # Download data
    df_raw = YahooDownloader(start_date=train_start, end_date=trade_end,
                             ticker_list=config_tickers.DOW_30_TICKER).fetch_data()
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=False,
        user_defined_feature=False
    )

    # Feature Engieering
    df_processed = fe.preprocess_data(df_raw)
    list_ticker = df_processed["tic"].unique().tolist()
    list_date = list(pd.date_range(df_processed['date'].min(), df_processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))
    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(df_processed, on=["date", "tic"],
                                                                              how="left")
    processed_full = processed_full[processed_full['date'].isin(df_processed['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])
    processed_full = processed_full.fillna(0)

    return processed_full


def format_data(data):
    # Load the stock price data from a CSV file
    # Convert 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'])
    return data


def calculate_slope(x, y):
    # Fit a linear regression model and return the slope
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    return model.coef_[0]


def categorize_trend(segment, slope_threshold_up, slope_threshold_down):
    # Determine the type of trend in the segment
    # x values are the number of days since the start of the segment
    x_values = np.array((segment['date'] - segment['date'].iloc[0]).dt.days)
    slope = calculate_slope(x_values, segment['close'].values)
    if slope > slope_threshold_up:  # slope_threshold is a predefined threshold for slope
        return "upward"
    elif slope < slope_threshold_down:
        return "downward"
    else:
        return "bumpy"


def analyze_trends(data, slope_threshold_up, slope_threshold_down):
    # Break the data into 3-month segments and analyze trends
    upward = []
    downward = []
    bumpy = []
    # Ensure the first date is a datetime object
    start_date = pd.to_datetime(data['date'].iloc[0])
    while start_date < pd.to_datetime(data['date'].iloc[-1]):
        # Determine the end date for the 3-month period
        end_date = start_date + pd.DateOffset(months=3)
        # Select the segment of data between start_date and end_date
        segment = data[(data['date'] >= start_date) & (data['date'] < end_date)]
        segment.set_index(pd.Index([*range(len(segment))]), inplace=True)
        if not segment.empty:
            trend = categorize_trend(segment, slope_threshold_up, slope_threshold_down)
            if trend == "upward":
                upward.append(segment)
            elif trend == "downward":
                downward.append(segment)
            else:  # bumpy trend
                bumpy.append(segment)
        # Update start_date to the next segment's start
        start_date = end_date

    return upward, downward, bumpy
