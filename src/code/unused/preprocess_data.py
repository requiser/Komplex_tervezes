import pandas as pd
import numpy as np

def process_stock_data(input_file, output_file):
    """
    Process stock data by calculating 'Tomorrow' and 'Target' columns, then save to a file.

    Parameters:
    - input_file: Path to the input CSV file.
    - output_file: Path to save the processed CSV file.
    """
    try:
        # Read the CSV file
        data = pd.read_csv(input_file)

        # Drop any columns containing 'unnamed' (in case of unnamed index columns)
        data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

        # Check for required columns
        required_columns = ["High", "Low"]
        if not all(column in data.columns for column in required_columns):
            raise ValueError(f"Missing required columns: {', '.join([col for col in required_columns if col not in data.columns])}")

        # Calculate 'Tomorrow' and 'Target' columns
        data["Tomorrow"] = (data["High"].shift(-1) + data["Low"].shift(-1)) / 2
        data["Target"] = (data["Tomorrow"] / ((data["High"] + data["Low"]) / 2) - 1)

        # Save the processed data to a CSV file
        data.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def add_rolling_average(df, sentiment_column, window=3):
    """
    Adds a rolling average column to smooth sentiment scores over a specified window.
    """
    df[f'{sentiment_column}_rolling_avg'] = df[sentiment_column].rolling(window=window).mean()
    return df

def categorize_sentiment(df, sentiment_column):
    """
    Categorizes sentiment scores into levels.
    """
    def categorize(score):
        if score > 0.6:
            return 'highly positive'
        elif 0.2 < score <= 0.6:
            return 'moderately positive'
        elif -0.2 <= score <= 0.2:
            return 'neutral'
        elif -0.6 <= score < -0.2:
            return 'moderately negative'
        else:
            return 'highly negative'

    df[f'{sentiment_column}_category'] = df[sentiment_column].apply(categorize)
    return df

def adjust_for_volatility(df, price_change_column):
    """
    Adjusts the stock price changes by dividing by the average daily volatility.
    """
    daily_volatility = df[price_change_column].std()  # Standard deviation as a measure of volatility
    df[f'{price_change_column}_volatility_adjusted'] = df[price_change_column] / daily_volatility
    return df

def sentiment_triggered_event_window(df, sentiment_column, threshold, window=3):
    """
    Calculates the average sentiment score within a specified window around rows
    where the sentiment score exceeds a threshold (positive or negative).
    """
    # Identify the rows where sentiment score crosses the positive or negative threshold
    trigger_indices = df[(df[sentiment_column] >= threshold) | (df[sentiment_column] <= -threshold)].index

    # Initialize lists to store average sentiment for pre and post window
    pre_event_scores = []
    post_event_scores = []

    # Calculate the rolling average sentiment score before and after each trigger
    for index in trigger_indices:
        # Pre-event sentiment window
        pre_window = df[sentiment_column].iloc[max(index - window, 0):index]
        pre_avg = pre_window.mean() if not pre_window.empty else None
        pre_event_scores.append(pre_avg)

        # Post-event sentiment window
        post_window = df[sentiment_column].iloc[index + 1:min(index + window + 1, len(df))]
        post_avg = post_window.mean() if not post_window.empty else None
        post_event_scores.append(post_avg)

    # Create a DataFrame to store the results and align with original DataFrame
    event_data = pd.DataFrame({
        'index': trigger_indices,
        f'{sentiment_column}_pre_event': pre_event_scores,
        f'{sentiment_column}_post_event': post_event_scores
    }).set_index('index')

    # Merge event data with the original DataFrame
    df = df.merge(event_data, left_index=True, right_index=True, how='left')

    return df

def calculate_volatility(df, window=20):
    """
    Calculate rolling volatility based on daily returns.

    Args:
        df (pd.DataFrame): DataFrame with historical stock data containing 'Date' and 'Close' columns.
        window (int): Rolling window in days for volatility calculation.

    Returns:
        pd.DataFrame: DataFrame with an additional 'Volatility' column.
    """
    # Calculate daily returns as percentage change
    df['Daily Return'] = df['Close'].pct_change()

    # Calculate rolling volatility (standard deviation of daily returns)
    df['Volatility'] = df['Daily Return'].rolling(window=window).std() * np.sqrt(252)  # Annualize volatility

    return df
