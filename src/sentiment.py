import glob
import pandas as pd
from transformers import pipeline
from src.logger import logger
from tqdm import tqdm

# Инициализация BERT
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def prepare_sentiment_data(SAVE_DIR):
    csv_files = glob.glob(os.path.join(SAVE_DIR, "*.csv"))
    df_list = [pd.read_csv(file) for file in csv_files]
    all_news_df = pd.concat(df_list, ignore_index=True)
    all_news_df['published_date'] = pd.to_datetime(all_news_df['published_date']).dt.date
    grouped_news = all_news_df.groupby('published_date').agg({
        'title': lambda titles: ' '.join(str(t) for t in titles if pd.notnull(t)),
        'snippet': lambda snippets: ' '.join(str(s) for s in snippets if pd.notnull(s)),
    }).reset_index()
    grouped_news['full_text'] = grouped_news['title'] + " " + grouped_news['snippet']
    return grouped_news

def analyze_sentiment_bert(text):
    if len(text) > 500:
        text = text[:500]
    result = sentiment_pipeline(text)[0]
    label = result['label'].lower()
    return label

def analyze_sentiment(grouped_news):
    tqdm.pandas()
    grouped_news['sentiment'] = grouped_news['full_text'].progress_apply(analyze_sentiment_bert)
    grouped_news[['published_date', 'sentiment', 'full_text']].to_csv("nyt_2024_daily_sentiment_bert.csv", index=False)
    logger.info(f"Процесс анализа завершён, данные сохранены.")
