import json
import logging
import os
import re
from datetime import datetime
from datetime import timedelta
from os import environ

import pandas as pd
import requests
import telegram
import yfinance as yf
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from django.conf import settings
from django.db.models import Avg
from django.db.models import Case
from django.db.models import Count
from django.db.models import ExpressionWrapper
from django.db.models import F
from django.db.models import FloatField
from django.db.models import Max
from django.db.models import Min
from django.db.models import When
from django.utils import timezone

from core.coinone import buy_ticker
from core.coinone import get_balances
from core.coinone import get_ticker
from core.db import ConnectionContextManager
from core.llm import invoke_llm
from kraken.models import Kraken

from .models import CryptoListing
from .models import Trade

# https://t.me/RichSebaBot
bot = telegram.Bot(environ["TELEGRAM_BOT_TOKEN"])
kraken = Kraken()

# Set timezone cache location to /tmp if AWS lambda environment
if not settings.DEBUG:
    yf.set_tz_cache_location("/tmp/yf")


class CryptoConfig(BaseModel):
    enabled: bool = Field(..., description="Whether this cryptocurrency is enabled for trading")
    min_amount: int = Field(..., description="Minimum amount in KRW to invest")
    max_amount: int = Field(..., description="Maximum amount in KRW to invest")
    step_amount: int = Field(..., description="Step amount in KRW for investment increments")

    @field_validator("min_amount", "max_amount", "step_amount")
    @classmethod
    def validate_amounts(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v

    @field_validator("max_amount")
    @classmethod
    def validate_max_amount(cls, v, info):
        min_amount = info.data.get("min_amount")
        if min_amount is not None and v < min_amount:
            raise ValueError("max_amount must be greater than min_amount")
        return v

    @field_validator("step_amount")
    @classmethod
    def validate_step_amount(cls, v, info):
        max_amount = info.data.get("max_amount")
        if max_amount is not None and v > max_amount:
            raise ValueError("step_amount must not be greater than max_amount")
        return v


CRYPTO_CONFIGS = {
    "BTC": CryptoConfig(
        enabled=False,  # 비활성화
        min_amount=5_000,
        max_amount=30_000,
        step_amount=5_000,
    ),
    "ETH": CryptoConfig(
        enabled=False,
        min_amount=5_000,
        max_amount=30_000,
        step_amount=5_000,
    ),
    "DOGE": CryptoConfig(
        enabled=True,
        min_amount=5_000,
        max_amount=30_000,
        step_amount=5_000,
    ),
    "SOL": CryptoConfig(
        enabled=True,
        min_amount=5_000,
        max_amount=30_000,
        step_amount=5_000,
    ),
    "DOT": CryptoConfig(
        enabled=True,
        min_amount=5_000,
        max_amount=30_000,
        step_amount=5_000,
    ),
    "FET": CryptoConfig(
        enabled=True,
        min_amount=5_000,
        max_amount=30_000,
        step_amount=5_000,
    ),
    "WLD": CryptoConfig(
        enabled=True,
        min_amount=5_000,
        max_amount=30_000,
        step_amount=5_000,
    ),
}


class BaseStrippedModel(BaseModel):
    def __init__(self, *args, **kwargs):
        kwargs = {k: v.strip() if isinstance(v, str) else v for k, v in kwargs.items()}
        super().__init__(*args, **kwargs)


class MultiCryptoRecommendation(BaseStrippedModel):
    scratchpad: str = Field(..., description="The analysis scratchpad text")
    reasoning: str = Field(..., description="The reasoning text")
    recommendations: list[dict] = Field(..., description="List of recommended cryptocurrencies and amounts")


def send_message(text, **kwargs):
    bot.sendMessage(chat_id=environ["TELEGRAM_BOT_CHANNEL_ID"], text=text, **kwargs)


def fetch_coinmarketcap_data(path, params=None):
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": os.getenv("COINMARKETCAP_API_KEY"),
    }
    url = "https://pro-api.coinmarketcap.com" + path

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    return response.json()["data"]


def collect_crypto_data(symbol: str, start_date: str):
    ticker = get_ticker(symbol)
    crypto_price = (float(ticker["best_asks"][0]["price"]) + float(ticker["best_bids"][0]["price"])) / 2

    crypto_data = fetch_coinmarketcap_data(
        "/v2/cryptocurrency/quotes/latest",
        params={"symbol": symbol, "convert": "KRW"},
    )[symbol][0]

    prev_balances = get_balances()
    crypto_balance = prev_balances.get(symbol) or {}

    input_data = dict(
        ticker,
        circulating_supply=crypto_data["circulating_supply"],
        max_supply=crypto_data["max_supply"],
        total_supply=crypto_data["total_supply"],
        **crypto_data["quote"]["KRW"],
        my_crypto_balance=crypto_balance,
        current_price=crypto_price,
    )

    # crypto data in KRW
    historical_data = fetch_crypto_data(symbol, "KRW", 30)
    df = pd.DataFrame(historical_data)
    df = df.drop(columns=["conversionType", "conversionSymbol"])
    crypto_data_csv = df.to_csv(index=False)

    # network stats (비트코인일 때만)
    network_stats_csv = ""
    if symbol == "BTC":
        network_stats = fetch_network_stats()
        df = pd.DataFrame(network_stats, index=[0])
        network_stats_csv = df.to_csv(index=False)

    # crypto news
    crypto_news = fetch_news(start_date, symbol)
    df = pd.DataFrame(crypto_news)
    df = df[["source", "title", "description", "publishedAt", "content"]]
    df["source"] = df["source"].apply(lambda x: x["name"])
    crypto_news_csv = df.to_csv(index=False)

    return {
        "symbol": symbol,
        "input_data": input_data,
        "crypto_data_csv": crypto_data_csv,
        "network_stats_csv": network_stats_csv,
        "crypto_news_csv": crypto_news_csv,
        "config": CRYPTO_CONFIGS[symbol],
    }


def get_multi_recommendation(crypto_data_list: list[dict], indices_csv: str) -> MultiCryptoRecommendation:
    # 각 코인별 데이터를 하나의 문자열로 조합
    data_descriptions = []
    for data in crypto_data_list:
        symbol = data["symbol"]
        description = f"""
=== {symbol} Data ===
Recent trading data in KRW in JSON (including user's current balance)
```json
{symbol}_json_data
```
Historical data in USD in CSV
```csv
{symbol}_crypto_data_csv
```"""

        if data["network_stats_csv"]:  # BTC인 경우
            description += f"""
Network stats in CSV
```csv
{symbol}_network_stats_csv
```"""

        description += f"""
News in CSV
```csv
{symbol}_crypto_news_csv
```"""
        description = re.sub(
            rf"^({symbol}_(json_data|crypto_data_csv|network_stats_csv|crypto_news_csv))",
            r"{\1}",
            description,
            flags=re.MULTILINE,
        )
        data_descriptions.append(description)

    all_data = "\n\n".join(data_descriptions)
    all_data += """
=== Market Indices ===
Indices data in USD in CSV
```csv
{indices_csv}
```"""

    # 각 코인의 설정 정보를 문자열로 조합
    config_descriptions = []
    for data in crypto_data_list:
        symbol = data["symbol"]
        config = data["config"]
        config_descriptions.append(
            f"{symbol}: min={config.min_amount:,}KRW, max={config.max_amount:,}KRW, step={config.step_amount:,}KRW"
        )

    # 각 코인별 데이터를 개별 변수로 전달하기 위한 kwargs 구성
    kwargs = {
        "indices_csv": indices_csv,
    }

    # 각 코인별로 데이터 변수 추가
    for data in crypto_data_list:
        symbol = data["symbol"]
        kwargs.update(
            {
                f"{symbol}_json_data": json.dumps(data["input_data"]),
                f"{symbol}_crypto_data_csv": data["crypto_data_csv"],
                f"{symbol}_crypto_news_csv": data["crypto_news_csv"],
            }
        )
        if data["network_stats_csv"]:  # BTC인 경우
            kwargs[f"{symbol}_network_stats_csv"] = data["network_stats_csv"]

    prompt = f"""You are an aggressive cryptocurrency investment advisor focusing on high-potential daily trading opportunities. You have access to real-time trading data, historical prices, market trends, and news for each cryptocurrency.

Available cryptocurrencies and their investment limits:
{chr(10).join(config_descriptions)}

Your task:
1. Analyze each cryptocurrency's data considering:
   - Short-term price movements and momentum
   - Trading volume spikes and trends
   - Recent news impact and market sentiment
   - Technical indicators and price patterns
   - Potential catalysts for price movement

2. Recommend 1-2 cryptocurrencies with the highest potential for short-term gains
   - MUST recommend at least 1 cryptocurrency regardless of market conditions
   - Focus on coins showing strong momentum or potential reversal signals
   - Look for opportunities in both upward and downward trends
   - Investment amount must be within specified limits for each coin
   - Amount should be in multiples of the step amount

The output should be in YAML format with these keys:
scratchpad: Detailed analysis of each cryptocurrency (in plain text, no markdown)
reasoning: Summary of your investment recommendations (in plain text, no markdown)
recommendations: List of selected cryptocurrencies and amounts

Example output:
```yaml
scratchpad: |
  DOGE 분석:
  - 현재가: 123.45 KRW (전일대비 -5%)
  - 거래량: 최근 4시간 동안 200% 급증
  - 모멘텀: RSI 과매도 구간, 반등 가능성
  - 뉴스: 새로운 개발 소식, 커뮤니티 활성화

  SOL 분석:
  - 현재가: 456.78 KRW (전일대비 +8%)
  - 거래량: 꾸준한 상승세, 전주 대비 80% 증가
  - 모멘텀: 상승 추세 지속 중, MACD 상향 돌파
  - 뉴스: DeFi 프로젝트 런칭 임박

reasoning: |
  DOGE: 과매도 상태에서 거래량 급증은 강력한 반등 신호. 커뮤니티 활동 증가로 단기 상승 가능성 높음
  SOL: 강한 상승 모멘텀과 함께 실질적인 개발 진전. 현재 추세가 이어질 것으로 예상

recommendations:
  - symbol: "DOGE"
    amount: 20000
  - symbol: "SOL"
    amount: 10000
```

Important notes:
1. Write analysis and reasoning in Korean
2. Use plain text format (no markdown syntax)
3. Focus on short-term trading signals and momentum
4. MUST recommend at least 1 coin even in bearish markets
5. Be aggressive but smart - look for strong technical setups
6. Consider news catalysts that could drive short-term movement
7. This analysis runs daily - focus on immediate opportunities"""

    return invoke_llm(
        MultiCryptoRecommendation,
        prompt,
        all_data,
        **kwargs,  # 각 코인별 데이터를 개별 변수로 전달
    )


def fetch_crypto_data(fsym, tsym, limit):
    url = f"https://min-api.cryptocompare.com/data/histoday"
    parameters = {"fsym": fsym, "tsym": tsym, "limit": limit}
    response = requests.get(url, params=parameters)
    data = response.json()
    return data["Data"]


def fetch_news(from_date, query):
    url = "https://newsapi.org/v2/everything"
    parameters = {"q": query, "from": from_date, "pageSize": 20}
    response = requests.get(
        url,
        params=parameters,
        headers={"X-Api-Key": os.getenv("NEWS_API_KEY")},
    )
    data = response.json()
    return data["articles"]


def fetch_network_stats():
    url = "https://api.blockchain.info/stats"
    response = requests.get(url)
    data = response.json()
    return data


def buy_crypto():
    # 오늘 날짜와 한 달 전 날짜 설정
    end_date = timezone.localdate()
    start_date = (end_date - timedelta(days=30)).strftime("%Y-%m-%d")

    # 유가, 다우존스, S&P 500, 나스닥
    indices = ["CL=F", "^DJI", "^GSPC", "^IXIC"]
    indices_data = yf.download(indices, start=start_date)
    indices_data_csv = indices_data["Close"].to_csv()

    # 활성화된 모든 코인의 데이터 수집
    crypto_data_list = []
    for symbol, config in CRYPTO_CONFIGS.items():
        if not config.enabled:
            continue
        try:
            crypto_data = collect_crypto_data(symbol, start_date)
            crypto_data_list.append(crypto_data)
        except Exception as e:
            logging.error(f"Failed to collect data for {symbol}: {e}")
            continue

    # LLM에게 추천 받기
    result, exc = [None] * 2
    for _ in range(3):
        try:
            result = get_multi_recommendation(crypto_data_list, indices_data_csv)
            break
        except Exception as e:
            logging.warning(e)
            exc = e

    if not result and exc:
        raise exc

    # 분석 결과 전송
    send_message(
        f"```\n코인 분석:\n{result.scratchpad}\n\n{result.reasoning}```",
        parse_mode=telegram.ParseMode.MARKDOWN_V2,
    )

    # 추천받은 코인 구매
    for recommendation in result.recommendations:
        symbol = recommendation["symbol"]
        amount = recommendation["amount"]

        if amount == 0:
            continue

        crypto_data = next(data for data in crypto_data_list if data["symbol"] == symbol)
        crypto_price = crypto_data["input_data"]["current_price"]
        crypto_balance = crypto_data["input_data"]["my_crypto_balance"]

        # buy crypto
        logging.info(f"{symbol} {crypto_price=}, {amount=}")

        # r = buy_ticker(symbol, amount)
        # logging.info(f"buy_ticker: {r}")

        # current balance and value after order
        balances = get_balances()
        crypto_amount = float(balances[symbol]["available"])
        crypto_value = crypto_amount * crypto_price
        krw_amount = float(balances["KRW"]["available"])
        bought_crypto = crypto_amount - float(crypto_balance.get("available") or 0)

        price_msg = "{:,.0f}".format(crypto_price)
        message_lines = [
            f"Buy: {bought_crypto:,.8f} {symbol} ({amount:,} KRW)",
            f"{crypto_amount:,.5f}{symbol} {crypto_value:,.0f} / {krw_amount:,.0f} KRW",
            f"{symbol} price: {price_msg}KRW",
        ]

        send_message("\n".join(message_lines))


# CoinMarketCap 메인 페이지 URL
COINMARKETCAP_URL = "https://coinmarketcap.com/"

# User-Agent 리스트
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.78.2 (KHTML, like Gecko) Version/7.0.6 Safari/537.78.2",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:78.0) Gecko/20100101 Firefox/78.0",
]


def fetch_crypto_listings():
    # https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyListingsLatest
    min_market_cap = 1_000
    params = {
        "convert": "USD",
        "cryptocurrency_type": "coins",
        "market_cap_min": min_market_cap,  # "Minimum market capitalization in USD"
        "limit": 1000,
    }
    data = fetch_coinmarketcap_data("/v1/cryptocurrency/listings/latest", params)

    listing = []
    for coin in data:
        quote = coin["quote"]["USD"]
        market_cap = quote["market_cap"]

        if market_cap >= min_market_cap:
            listing.append(
                CryptoListing(
                    name=coin["name"],
                    symbol=coin["symbol"],
                    data_at=coin["last_updated"],
                    rank=coin["cmc_rank"],
                    circulating_supply=coin["circulating_supply"],
                    total_supply=coin["total_supply"],
                    max_supply=coin["max_supply"],
                    price=quote["price"],
                    market_cap=market_cap,
                    change_1h=quote["percent_change_1h"],
                    change_24h=quote["percent_change_24h"],
                    change_7d=quote["percent_change_7d"],
                    volume_24h=quote["volume_24h"],
                    raw=coin,
                )
            )

    with ConnectionContextManager():
        result = CryptoListing.objects.bulk_create(listing)
        logging.info(f"fetch_crypto_listings: {len(result)}")


def pretty_currency(value):
    if value > 1_000_000_000_000:
        value = f"{value/1_000_000_000_000:,.2f}T"
    elif value > 1_000_000_000:
        value = f"{value/1_000_000_000:,.2f}B"
    elif value > 1_000_000:
        value = f"{value/1_000_000:,.2f}M"
    elif value > 1_000:
        value = f"{value/1_000:,.2f}K"

    return value


def select_coins_to_buy():
    today = timezone.now().date()
    start_date = today - timedelta(days=4)

    # 최근 5일 동안 24시간 변동률이 모두 0.5% 이상인 코인을 선택하고 필요한 정보를 한번에 가져옴
    coins = (
        CryptoListing.objects.filter(
            data_at__date__range=(start_date, today),
            market_cap__gt=10_000_000,
            volume_24h__gt=100_000,
            rank__lt=300,
        )
        .values("symbol")
        .annotate(
            count_positive=Count(Case(When(change_24h__gte=0.5, then=1))),
            first_price=Min("price"),
            last_price=Max("price"),
            avg_market_cap=Avg("market_cap"),
            name=F("name"),
        )
        .filter(count_positive=5)
        .annotate(
            change_5d=ExpressionWrapper(
                (F("last_price") - F("first_price")) / F("first_price") * 100,
                output_field=FloatField(),
            )
        )
        .order_by("-change_5d", "-avg_market_cap")[:10]
    )

    text_list = []

    # 선택된 코인 정보 출력
    for i, coin in enumerate(coins, 1):
        text_list.extend([f"{i}. {coin['name']} ({coin['symbol']}) ${coin['last_price']:.4f}"])
        text_list.append(f"Price 5 days ago: ${coin['first_price']:.4f}")
        text_list.append(f"Change over 5 days: {coin['change_5d']:.2f}%")

        market_cap = pretty_currency(coin["avg_market_cap"])
        text_list.extend([f"Average Market Cap: {market_cap}", ""])

    if text_list:
        text = "\n".join(text_list)
        text = f"Selected Coins to Buy:\n```\n{text}```"
    else:
        text = "No coins met the criteria for buying\."

    send_message(text, parse_mode=telegram.ParseMode.MARKDOWN_V2)


def insert_trade_history():
    try:
        trade = Trade.objects.earliest("trade_at")
        end = trade.trade_at - timedelta(seconds=1)
    except Trade.DoesNotExist:
        end = timezone.now()

    days = 365
    start = end - timedelta(days=days)
    trades = []

    while True:
        logging.info(f"{start=}, {end=}")
        start_ts = start.timestamp()
        df = kraken.get_trades(start=start_ts, end=end.timestamp())[0]

        # row count
        if df.shape[0] == 0:
            break

        end_ts = df["time"][-1] - 1
        if end_ts <= start_ts:
            break

        end = datetime.fromtimestamp(end_ts)

        for _, row in df.iterrows():
            trades.append(
                Trade(
                    txid=row["txid"],
                    pair=row["pair"],
                    trade_at=timezone.make_aware(datetime.fromtimestamp(row["time"])),
                    order_type=row["type"],
                    price=row["price"],
                    cost=row["cost"],
                    volume=row["vol"],
                    fee=row["fee"],
                    margin=row["margin"],
                    misc=row["misc"],
                    raw=row.to_dict(),
                )
            )

    trades.sort(key=lambda x: x.trade_at)
    Trade.objects.bulk_create(trades)


def update_trade_history():
    with ConnectionContextManager():
        trade = Trade.objects.latest("trade_at")
        start = trade.trade_at + timedelta(seconds=1)

        df = kraken.get_trades(start=start.timestamp())[0]
        trades = [
            Trade(
                txid=row["txid"],
                pair=row["pair"],
                trade_at=timezone.make_aware(datetime.fromtimestamp(row["time"])),
                order_type=row["type"],
                price=row["price"],
                cost=row["cost"],
                volume=row["vol"],
                fee=row["fee"],
                margin=row["margin"],
                misc=row["misc"],
                raw=row.to_dict(),
            )
            for _, row in df.iterrows()
        ]

        trades.sort(key=lambda x: x.trade_at)
        logging.info(f"update_trade_history: {Trade.objects.bulk_create(trades)}")
