import yfinance as yf
from src.config import TICKERS, START_DATE, END_DATE

def download_financial_data():
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, group_by='ticker')
    data.to_csv("data/financial_data.csv")
    return data
