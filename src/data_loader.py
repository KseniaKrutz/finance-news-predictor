import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from src.config import API_KEY, SAVE_DIR
from src.logger import logger

def download_month(month):
    url = f"https://api.nytimes.com/svc/archive/v1/2024/{month}.json?api-key={API_KEY}"
    response = requests.get(url)
    articles = []
    if response.status_code == 200:
        data = response.json()
        for doc in data['response']['docs']:
            articles.append({
                "title": doc.get('headline', {}).get('main'),
                "url": doc.get('web_url'),
                "published_date": doc.get('pub_date'),
                "section": doc.get('section_name') or 'Unknown',
                "snippet": doc.get('snippet'),
                "source": doc.get('source')
            })
        logger.info(f"Загружено {len(articles)} статей за месяц {month}")
    else:
        logger.error(f"Ошибка загрузки месяца {month}: {response.status_code}")
    return articles

def save_articles_by_section(all_articles):
    df = pd.DataFrame(all_articles)
    grouped = df.groupby('section')
    for section, group in grouped:
        safe_section = "".join(c if c.isalnum() else "_" for c in (section or "Unknown"))
        filename = os.path.join(SAVE_DIR, f"{safe_section}.csv")
        group.to_csv(filename, index=False)
        logger.info(f"Сохранено {len(group)} новостей в раздел '{section}'")

def download_all_articles():
    all_articles = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_month, month) for month in range(1, 13)]
        for future in futures:
            all_articles.extend(future.result())
    save_articles_by_section(all_articles)
    return all_articles
