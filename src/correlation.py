import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import TICKERS

def correlation_analysis(daily_sentiment, financial_data):
    volume_data = financial_data.xs('Volume', axis=1, level=1)
    volume_data.index = pd.to_datetime(volume_data.index)
    volume_data = volume_data.reset_index()
    merged_data = pd.merge(daily_sentiment, volume_data, left_on='published_date', right_on='Date', how='inner')
    correlation_matrix = merged_data.drop(columns=['published_date', 'Date']).corr()
    plt.figure(figsize=(14,10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Корреляция между количеством новостей и объемом торгов по дням")
    plt.show()
    logger.info("Корреляция и графики завершены.")
