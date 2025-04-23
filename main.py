from src.data_loader import download_all_articles
from src.financial_data import download_financial_data
from src.sentiment import prepare_sentiment_data, analyze_sentiment
from src.lstm_train import lstm_train
from src.transformer_train import transformer_train
from src.logger import logger

if __name__ == "__main__":
    logger.info("🚀 Запуск проекта")
    download_all_articles()
    financial_data = download_financial_data()
    grouped_news = prepare_sentiment_data()
    analyze_sentiment(grouped_news)
    lstm_model = lstm_train(grouped_news)
    transformer_model = transformer_train(grouped_news)
    logger.info("Процесс обучения завершён")
