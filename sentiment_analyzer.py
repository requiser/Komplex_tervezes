import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from scipy.special import softmax
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

# import nltk

plt.style.use('ggplot')

sia = SentimentIntensityAnalyzer()

MODEL = f"mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

sent_pipeline = pipeline("sentiment-analysis",
                         model=model,
                         tokenizer=tokenizer)


def polarity_scores_roberta(example):
    example = example[:514]
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


def get_results(df, news, debug=False):
    res = {}
    fail = {}
    n = 0
    for i, row in tqdm(df.iterrows(),
                       total=len(df)):
        text = row[news]
        try:
            if pd.isna(text):
                roberta_result = {
                    'roberta_neg': 0,
                    'roberta_neu': 1,
                    'roberta_pos': 0
                }

            else:
                roberta_result = polarity_scores_roberta(text)

        except (IndexError, RuntimeError):
            if pd.isna(text) & debug:
                fail[n] = i.astype(str)
            elif debug:
                fail[n] = text
                n += 1
            roberta_result = {
                'roberta_neg': 0,
                'roberta_neu': 0,
                'roberta_pos': 0
            }
            pass
        res[i] = roberta_result

    res = pd.DataFrame(res).T

    res['sentiment score'] = res.apply(
        lambda x: -1
        if max(x['roberta_neg'],
               x['roberta_pos'],
               x['roberta_neu']) == x['roberta_neg']
        else (1 if max(x['roberta_neg'],
                       x['roberta_pos'],
                       x['roberta_neu']) == x['roberta_pos']
              else 0) * max(x['roberta_neg'], x['roberta_pos'], x['roberta_neu']), axis=1
    )
    res = pd.merge(df, res['sentiment score'], left_index=True, right_index=True, suffixes=('_original', ''))

    if debug:
        return res, fail
    else:
        return res


def check_prediction(df, tar, sco, neutral, debug=False):
    predict_dict = {}
    fail = {}
    n = 0
    for i, row in tqdm(df.iterrows(),
                       total=len(df)):
        try:
            target = row[tar]
            score = row[sco]
            if abs(target) > neutral:
                if (target / score) > 0:
                    predict_dict[i] = 1
                elif (target / score) < 0:
                    predict_dict[i] = 0
            elif score == 0:
                predict_dict[i] = 1
            else:
                predict_dict[i] = 0
        except KeyError:
            predict_dict[i] = 'null'
        except:
            if debug:
                fail[n] = i
                n += 1
            predict_dict[i] = 0
            pass
    predictions = pd.DataFrame(predict_dict, index=[0]).T
    predictions.columns = ['predictions']
    predictions = pd.merge(df, predictions['predictions'], left_index=True, right_index=True)
    if debug:
        return predictions, fail
    else:
        return predictions


def get_results_with_pipe(df, news, debug=False):  # don't use it in current form
    res = {}
    fail = {}
    n = 0
    for i, row in tqdm(df.iterrows(),
                       total=len(df)):
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
            if pd.isna(text) & debug:
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

    res = pd.DataFrame(res).T
    res = res[0].apply(pd.Series)
    res.columns = ['label', 'score']

    res['pipe sentiment score'] = res.apply(
        lambda x: -1 if x['label'] == 'negative' else (1 if x['label'] == 'positive' else 0) * x['score'],
        axis=1
    )
    res = pd.merge(df, res['pipe sentiment score'], left_index=True, right_index=True, suffixes=('_original', ''))

    if debug:
        return res, fail
    else:
        return res


def check_pipe_prediction(df, tar, label, debug=False):  # don't use it in current form
    predict_dict = {}
    fail = {}
    n = 0
    for i, row in tqdm(df.iterrows(),
                       total=len(df)):
        try:
            target = row[tar]
            label = row[label]
            match label:
                case 'negative':
                    predict_dict[i] = int(target == 0)
                case 'positive':
                    predict_dict[i] = int(target == 1)
                case 'neutral':
                    predict_dict[i] = int(target == df[tar][i - 1])
        except KeyError:
            predict_dict[i] = 0
        except:
            if debug:
                fail[n] = i
                n += 1
            predict_dict[i] = "null"
            pass
    if debug:
        return predict_dict, fail
    else:
        return predict_dict
