import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from scipy.special import softmax
from tqdm.notebook import tqdm
import nltk
import nltk.sentiment.util
from nltk.sentiment.vader import SentimentIntensityAnalyzer

plt.style.use('ggplot')

MODEL = f"mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

_sentiment_analysis_mrm8488 = pipeline("sentiment-analysis",
                                       model=model,
                                       tokenizer=tokenizer)

_sentiment_analysis_ProsusAI = pipeline("sentiment-analysis",
                                       model="ProsusAI/finbert")


def check_prediction(df, tar, sco, model, neutral_factor=1, debug=False):
    """
    Check prediction based on adaptive volatility-based threshold for neutrality.

    Args:
        df (pd.DataFrame): DataFrame containing stock data with calculated volatility.
        tar (str): Column name for the target price change (e.g., 'Daily Return').
        sco (str): Column name for the sentiment score.
        model (str): Model name to include in the predictions column.
        neutral_factor (float): Multiplier for volatility to define the neutral threshold.
        debug (bool): If True, logs details of failures.

    Returns:
        pd.DataFrame: DataFrame with predictions and success/failure of predictions.
    """
    predict_dict = {}
    fail = {}
    n = 0

    # Calculate adaptive volatility threshold and predictions
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            target = row[tar]
            score = row[sco]
            volatility_threshold = row['Volatility'] * neutral_factor  # Adaptive neutral threshold

            if abs(target) > volatility_threshold:
                predict_dict[i] = 1 if (target / score) > 0 else 0  # Correct or incorrect prediction
            else:
                predict_dict[i] = 1 if score == 0 else 0  # Neutral prediction based on sentiment score

        except KeyError:
            predict_dict[i] = 'null'
        except:
            if debug:
                fail[n] = i
                n += 1
            predict_dict[i] = 0
            pass

    # Create the predictions DataFrame and replace the existing column if necessary
    predictions = pd.DataFrame(predict_dict, index=[0]).T
    predictions_col = f"{model}_predictions"
    predictions.columns = [predictions_col]
    if predictions_col in df.columns:
        df = df.drop(columns=[predictions_col])
    df = pd.merge(df, predictions, left_index=True, right_index=True)

    if debug:
        return df, fail
    else:
        return df


def get_results(df, news, model, debug=False):
    """
    Performs sentiment analysis using a specified pipeline and calculates sentiment scores.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        news (str): Column name for the news headlines or text to analyze.
        model (str): The model to use for sentiment analysis ('mrm8488' or 'ProsusAI').
        debug (bool): If True, logs details of failures.

    Returns:
        pd.DataFrame: DataFrame with calculated sentiment scores.
    """
    # Select the appropriate model pipeline
    if model == "mrm8488":
        sent_pipeline = _sentiment_analysis_mrm8488
    elif model == "ProsusAI":
        sent_pipeline = _sentiment_analysis_ProsusAI
    else:
        raise ValueError("Invalid model name. Choose 'mrm8488' or 'ProsusAI'.")

    res = {}
    fail = {}
    n = 0

    # Process each row in the DataFrame
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row[news]
        try:
            if pd.isna(text):
                roberta_result = [{
                    'label': 'neutral',
                    'score': 1
                }]
            else:
                roberta_result = sent_pipeline(text[:514])

        except (IndexError, RuntimeError):
            if pd.isna(text) and debug:
                fail[n] = str(i)
            elif debug:
                fail[n] = text
                n += 1
            roberta_result = [{
                'label': 'neutral',
                'score': 0
            }]
            pass
        res[i] = roberta_result

    # Convert results to DataFrame
    res = pd.DataFrame(res).T
    res = res[0].apply(pd.Series)
    res.columns = ['label', 'score']

    # Calculate sentiment score with intensity and replace if column exists
    sentiment_score_col = f"{model}_sentiment_score"
    res[sentiment_score_col] = res.apply(
        lambda x: -x['score'] if x['label'] == 'negative'
        else (x['score'] if x['label'] == 'positive' else 0),
        axis=1
    )

    # Merge with original DataFrame, dropping the column if it already exists
    if sentiment_score_col in df.columns:
        df = df.drop(columns=[sentiment_score_col])
    df = pd.merge(df, res[sentiment_score_col], left_index=True, right_index=True)

    if debug:
        return df, fail
    else:
        return df

