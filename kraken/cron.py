import json
import logging
import os
from datetime import datetime
from datetime import timedelta
from os import environ

import pandas as pd
import requests
import telegram
import yfinance as yf
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.pydantic_v1 import Field
from langchain_core.pydantic_v1 import validator

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

    @validator("min_amount", "max_amount", "step_amount")
    def validate_amounts(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v

    @validator("max_amount")
    def validate_max_amount(cls, v, values):
        if "min_amount" in values and v < values["min_amount"]:
            raise ValueError("max_amount must be greater than min_amount")
        return v

    @validator("step_amount")
    def validate_step_amount(cls, v, values):
        if "max_amount" in values and v > values["max_amount"]:
            raise ValueError("step_amount must not be greater than max_amount")
        return v


CRYPTO_CONFIGS = {
    "BTC": CryptoConfig(
        enabled=False,  # 비트코인 비활성화
        min_amount=5_000,
        max_amount=30_000,
        step_amount=5_000,
    ),
    "ETH": CryptoConfig(
        enabled=True,
        min_amount=10_000,
        max_amount=30_000,
        step_amount=5_000,
    ),
    "DOGE": CryptoConfig(
        enabled=False,
        min_amount=5_000,
        max_amount=20_000,
        step_amount=5_000,
    ),
    "DOT": CryptoConfig(
        enabled=True,
        min_amount=10_000,
        max_amount=30_000,
        step_amount=5_000,
    ),
}


class BaseStrippedModel(BaseModel):
    def __init__(self, *args, **kwargs):
        kwargs = {k: v.strip() if isinstance(v, str) else v for k, v in kwargs.items()}
        super().__init__(*args, **kwargs)


class InvestRecommendation(BaseStrippedModel):
    scratchpad: str = Field(..., description="The scratchpad text")
    reasoning: str = Field(..., description="The reasoning text")
    amount: int = Field(..., description="The amount in KRW to invest")
    symbol: str = Field(..., description="The cryptocurrency symbol")


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


def get_recommendation(
    symbol: str, data, crypto_data_csv, network_stats_csv, indices_csv, crypto_news_csv
) -> InvestRecommendation:
    json_data = json.dumps(data)
    config = CRYPTO_CONFIGS[symbol]
    amount_min = config.min_amount
    amount_max = config.max_amount
    amount_step = config.step_amount

    if symbol == "BTC":
        network_stats = f"""
Network stats in CSV
```csv
{network_stats_csv}
```
""".strip()
    else:
        network_stats = ""

    data_description = """
Recent {symbol} trading data in KRW in JSON (including user's current {symbol} balance)
```json
{json_data}
```
{symbol} data in USD in CSV
```csv
{crypto_data_csv}
```
{network_stats}
Indices data in CSV
```csv
{indices_csv}
```
{symbol} news in CSV
```csv
{crypto_news_csv}
```
""".strip()

    prompt = f"""
You are a cryptocurrency investment advisor. You will be provided with recent {symbol} trading data in JSON format (including the user's current {symbol} balance) and other data in CSV format.
Your task is to analyze the data and recommend a KRW amount to purchase {symbol} worth between {amount_min} and {amount_max} in multiples of {amount_step} at the same time every day.
If you don't think it's a good time to purchase any {symbol}, output 0.

Consider the user's current {symbol} balance when making your recommendation. If the user already has a significant amount of {symbol}, you may want to recommend a lower purchase amount or no purchase at all.

Based on the data, what amount of KRW between {amount_min} and {amount_max} (in multiples of {amount_step}) would you recommend purchasing {symbol} at the same time every day? If you don't recommend any purchase, output 0.

The output should be in YAML format and keys should be `scratchpad`, `reasoning`, `amount`, and `symbol`.

Output example:
```yaml
scratchpad: |
  [여기에 최근 {symbol} 데이터의 분석과 생각 과정을 한국어로 간단히 적으세요.]

reasoning: |
  [분석을 기반으로 권장하는 구매 금액에 대한 주요 이유를 한국어로 간단히 요약하세요.]

amount: |
  [추천하는 구매 금액을 Integer로 입력하세요]

symbol: "{symbol}"
```""".strip()

    return invoke_llm(
        InvestRecommendation,
        prompt,
        data_description,
        symbol=symbol,
        json_data=json_data,
        crypto_data_csv=crypto_data_csv,
        network_stats=network_stats,
        indices_csv=indices_csv,
        crypto_news_csv=crypto_news_csv,
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

    for symbol, config in CRYPTO_CONFIGS.items():
        if not config.enabled:
            continue

        ticker = get_ticker(symbol)

        # average of ask, bid
        crypto_price = (float(ticker["best_asks"][0]["price"]) + float(ticker["best_bids"][0]["price"])) / 2

        crypto_data = fetch_coinmarketcap_data(
            "/v2/cryptocurrency/quotes/latest",
            params={"symbol": symbol, "convert": "KRW"},
        )[symbol][0]

        prev_balances = get_balances()

        input = dict(
            ticker,
            circulating_supply=crypto_data["circulating_supply"],
            max_supply=crypto_data["max_supply"],
            total_supply=crypto_data["total_supply"],
            **crypto_data["quote"]["KRW"],
            my_crypto_balance=prev_balances[symbol],
        )

        # crypto data in KRW
        crypto_data = fetch_crypto_data(symbol, "KRW", 30)
        df = pd.DataFrame(crypto_data)
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

        # get recommendation
        result, exc = [None] * 2
        for _ in range(3):
            try:
                result = get_recommendation(
                    symbol, input, crypto_data_csv, network_stats_csv, indices_data_csv, crypto_news_csv
                )
                break
            except Exception as e:
                logging.warning(e)
                exc = e

        if not result and exc:
            raise exc

        send_message(
            f"```\n{symbol} 분석:\n{result.scratchpad}\n\n{result.reasoning}```",
            parse_mode=telegram.ParseMode.MARKDOWN_V2,
        )

        if result.amount == 0:
            continue

        # buy crypto by recommended amount
        logging.info(f"{symbol} {crypto_price=}, {result.amount=}")

        r = buy_ticker(symbol, result.amount)
        logging.info(f"buy_ticker: {r}")

        # current balance and value after order
        balances = get_balances()
        crypto_amount = float(balances[symbol]["available"])
        crypto_value = crypto_amount * crypto_price
        krw_amount = float(balances["KRW"]["available"])
        bought_crypto = crypto_amount - float(prev_balances[symbol]["available"])

        price_msg = "{:,.0f}".format(crypto_price)
        message_lines = [
            f"Buy: {bought_crypto:,.8f} {symbol} ({result.amount:,} KRW)",
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
