import json
import logging
import os
from datetime import datetime
from datetime import timedelta
from os import environ

import requests
import telegram
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.pydantic_v1 import Field
from pykrakenapi.pykrakenapi import KrakenAPIError

from django.utils import timezone

from core.llm import invoke_llm
from kraken.models import Kraken

from .models import Trade

bot = telegram.Bot(environ["TELEGRAM_BOT_TOKEN"])
kraken = Kraken()


class BaseStrippedModel(BaseModel):
    def __init__(self, *args, **kwargs):
        kwargs = {k: v.strip() if isinstance(v, str) else v for k, v in kwargs.items()}
        super().__init__(*args, **kwargs)


class InvestRecommendation(BaseStrippedModel):
    scratchpad: str = Field(..., description="The scratchpad text")
    reasoning: str = Field(..., description="The reasoning text")
    amount: int = Field(..., description="The amount in Euro to invest")


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


def get_recommendation(data) -> InvestRecommendation:
    json_data = json.dumps(data)
    prompt = """
You are a Bitcoin investment advisor. You will be provided with recent Bitcoin trading data in JSON format and the numerical data is presented in euros. Your task is to analyze the data and recommend a Euro amount to purchase Bitcoin worth between €5 and €30 in multiples of €5 (e.g., €5, €10, €15, ..., €30) at the same time every day. If you don't think it's a good time to purchase any Bitcoin, output 0.

Key explanations:

a = ask array(<price>, <whole lot volume>, <lot volume>)
b = bid array(<price>, <whole lot volume>, <lot volume>)
c = last trade closed array(<price>, <lot volume>)
v = volume array(<today>, <last 24 hours>)
p = volume weighted average price array(<today>, <last 24 hours>)
t = number of trades array(<today>, <last 24 hours>)
l = low array(<today>, <last 24 hours>)
h = high array(<today>, <last 24 hours>)
o = today's opening price

Based on the data, what amount of Euros between €5 and €30 (in multiples of €5) would you recommend purchasing Bitcoin at the same time every day? If you don't recommend any purchase, output 0.

The output should be in YAML format and keys should be `scratchpad`, `reasoning`, and `amount`.

Output example:
```yaml
scratchpad: |
  [여기에 최근 비트코인 데이터의 분석과 생각 과정을 한국어로 간단히 적으세요. 현재 가격이 개장 가격 및 일일 고점/저점 대비 어떤지, 거래량, 전반적인 추세 등을 고려하세요.]

reasoning: |
  [분석을 기반으로 권장하는 구매 금액에 대한 주요 이유를 한국어로 간단히 요약하세요. 왜 그 금액을 구매하는 것이 좋다고 생각하는지 또는 왜 구매를 권장하지 않는지 설명하세요.]

amount: |
  [추천하는 구매 금액을 Integer로 입력하세요. €5에서 €30 사이의 금액이어야 하며, €5의 배수여야 합니다.]
```""".strip()
    return invoke_llm(
        InvestRecommendation,
        prompt,
        "Recent Bitcoin trading data in JSON\n{json_data}",
        json_data=json_data,
    )


def buy_bitcoin():
    pair = "XXBTZEUR"
    ticker = kraken.get_ticker(pair)
    ticker = {k: v[pair] for k, v in ticker.items()}

    # average of ask, bid
    btc_price = (float(ticker["a"][0]) + float(ticker["b"][0])) / 2

    balance = kraken.get_account_balance()

    btc_data = fetch_coinmarketcap_data(
        "/v2/cryptocurrency/quotes/latest",
        params={"symbol": "BTC", "convert": "EUR"},
    )["BTC"][0]

    input = dict(
        ticker,
        circulating_supply=btc_data["circulating_supply"],
        max_supply=btc_data["max_supply"],
        total_supply=btc_data["total_supply"],
        **btc_data["quote"]["EUR"],
    )

    # get recommendation
    result, exc = [None] * 2
    for _ in range(3):
        try:
            result = get_recommendation(input)
            break
        except Exception as e:
            logging.warning(e)
            exc = e

    if not result and exc:
        raise exc

    send_message(
        f"```\n{result.scratchpad}\n\n{result.reasoning}```",
        parse_mode=telegram.ParseMode.MARKDOWN_V2,
    )

    if result.amount == 0:
        return

    # buy Bitcoin by recommended amount
    amount = result.amount / btc_price
    logging.info(f"{btc_price=}, {amount=}")

    is_test = False
    start = timezone.now()

    while True:
        try:
            r = kraken.api.add_standard_order(
                pair=pair, type="buy", ordertype="market", volume=amount, validate=is_test
            )
            logging.info(f"add_standard_order: {r}")
        except KrakenAPIError as e:
            error = str(e)
            if "EService:" in error:
                logging.warning(error)
                continue
            else:
                raise e

        break

    # current balance and value after order
    balance = kraken.get_account_balance()
    btc_amount = balance["XXBT"]["amount"]
    btc_value = btc_amount * btc_price
    euro_amount = balance["ZEUR"]["amount"]

    send_message(
        "\n".join(
            [
                f"Buy: {amount:,.8f} BTC ({result.amount}€)",
                f"{btc_amount:,.5f}BTC {btc_value:,.2f}€ / {euro_amount:,.2f}€",
                "BTC price: {:,.2f}€".format(btc_price),
            ]
        )
    )

    try:
        # save trade history
        df = kraken.get_trades(start=start.timestamp(), end=timezone.now().timestamp())[0]
        if df.shape[0] > 0:
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
            ret = Trade.objects.bulk_create(trades)
            logging.info(f"Trade created: {len(ret)}")
    except Exception as e:
        logging.warning(e)


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


# 현재 코인 데이터 스크래핑 함수
def fetch_current_data():
    # https://coinmarketcap.com/api/documentation/v1/#operation/getV1CryptocurrencyListingsLatest
    params = {
        "convert": "USD",
        "cryptocurrency_type": "coins",
    }
    data = fetch_coinmarketcap_data("/v1/cryptocurrency/listings/latest", params)

    coins = []
    for coin in data:
        name = coin["name"]
        symbol = coin["symbol"]
        quote = coin["quote"]["USD"]
        price = quote["price"]
        change_1h = quote["percent_change_1h"]
        change_24h = quote["percent_change_24h"]
        change_7d = quote["percent_change_7d"]
        market_cap = quote["market_cap"]
        volumne_24h = quote["volume_24h"]

        coins.append(
            {
                "name": name,
                "symbol": symbol,
                "price": price,
                "change_7d": change_7d,
                "change_24h": change_24h,
                "change_1h": change_1h,
                "market_cap": market_cap,
                "volume_24h": volumne_24h,
            }
        )

    return coins


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


# 코인 선택 로직 구현
def select_coins_to_buy():
    # 코인 데이터 스크래핑
    coins = fetch_current_data()

    # 7일 변동률이 7% 이상이고 24시간 변동률이 양수인 코인을 선택
    sorted_coins = sorted(
        filter(
            lambda x: x["change_7d"] >= 7
            and x["change_24h"] > 0
            and x["market_cap"] > 10_000_000
            and x["volume_24h"] > 100_000,
            coins,
        ),
        key=lambda x: (x["change_7d"], x["change_24h"]),
        reverse=True,
    )
    selected_coins = sorted_coins[:10]  # 상위 코인 선택

    text_list = []

    # 선택된 코인 정보 출력
    for i, coin in enumerate(selected_coins, 1):
        text_list.extend([f"{i}. {coin['name']} ({coin['symbol']})", f"Price: ${coin['price']:.4f}"])

        values = "/".join(f"{val:.1f}%" for val in (coin["change_7d"], coin["change_24h"], coin["change_1h"]))
        text_list.append(f"Change(7d/24/1h): {values}")

        market_cap, volume = [pretty_currency(val) for val in (coin["market_cap"], coin["volume_24h"])]
        text_list.extend([f"Market Cap: {market_cap}", f"Volume(24h): {volume}", ""])

    text = "\n".join(text_list)
    send_message(
        f"Selected Coins to Buy:\n```\n{text}```",
        parse_mode=telegram.ParseMode.MARKDOWN_V2,
    )


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
