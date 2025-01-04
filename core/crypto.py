import os

import requests
import yfinance as yf

from django.conf import settings


def fetch_coinmarketcap_data(path, params=None):
    """CoinMarketCap API를 호출하여 데이터를 가져옵니다."""
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": os.getenv("COINMARKETCAP_API_KEY"),
    }
    url = "https://pro-api.coinmarketcap.com" + path

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    return response.json()["data"]


def get_latest_listings(min_market_cap=1_000, limit=1000):
    """최신 암호화폐 목록을 가져옵니다."""
    params = {
        "convert": "USD",
        "cryptocurrency_type": "coins",
        "market_cap_min": min_market_cap,
        "limit": limit,
    }
    return fetch_coinmarketcap_data("/v1/cryptocurrency/listings/latest", params)


def get_quotes(symbol):
    """특정 암호화폐의 최신 시세 정보를 가져옵니다."""
    return fetch_coinmarketcap_data(
        "/v2/cryptocurrency/quotes/latest",
        params={"symbol": symbol, "convert": "KRW"},
    )[
        symbol
    ][0]


def get_historical_data(fsym, tsym, limit):
    """특정 암호화폐의 과거 데이터를 가져옵니다."""
    url = "https://min-api.cryptocompare.com/data/histoday"
    parameters = {"fsym": fsym, "tsym": tsym, "limit": limit}
    response = requests.get(url, params=parameters)
    data = response.json()
    return data["Data"]


def get_network_stats():
    """블록체인 네트워크 통계를 가져옵니다."""
    url = "https://api.blockchain.info/stats"
    response = requests.get(url)
    return response.json()


def fetch_news(from_date, query):
    """뉴스 API를 통해 암호화폐 관련 뉴스를 가져옵니다."""
    url = "https://newsapi.org/v2/everything"
    parameters = {"q": query, "from": from_date, "pageSize": 10}
    response = requests.get(
        url,
        params=parameters,
        headers={"X-Api-Key": os.getenv("NEWS_API_KEY")},
    )
    data = response.json()
    return data["articles"]


def get_market_indices(start_date):
    """주요 시장 지표 데이터를 가져옵니다."""
    if not settings.DEBUG:
        yf.set_tz_cache_location("/tmp/yf")

    indices = ["CL=F", "^DJI", "^GSPC", "^IXIC"]
    indices_data = yf.download(indices, start=start_date)
    return indices_data["Close"].to_csv()
