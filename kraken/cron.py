import logging
import os
from os import environ

import requests
import telegram
from pykrakenapi.pykrakenapi import KrakenAPIError

from kraken.models import Kraken

bot = telegram.Bot(environ["TELEGRAM_BOT_TOKEN"])


def send_message(text, **kwargs):
    bot.sendMessage(chat_id=environ["TELEGRAM_BOT_CHANNEL_ID"], text=text, **kwargs)


def buy_bitcoin():
    kraken = Kraken()

    pair = "XXBTZEUR"
    ticker = kraken.get_ticker(pair)

    # average of ask, bid
    btc_price = (float(ticker["a"][pair][0]) + float(ticker["b"][pair][0])) / 2

    # buy Bitcoin by 10 euros
    amount = 10 / btc_price
    logging.info(f"{btc_price=}, {amount=}")

    is_test = False

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
                f"Buy: {amount:,.8f} BTC",
                f"{btc_amount:,.5f}BTC {btc_value:,.2f}€ / {euro_amount:,.2f}€",
                "BTC price: {:,.2f}€".format(btc_price),
            ]
        )
    )


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
    URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": os.getenv("COINMARKETCAP_API_KEY"),
    }
    params = {
        "start": "1",
        "limit": "100",
        "convert": "USD",
        "cryptocurrency_type": "coins",
    }

    # API 호출
    response = requests.get(URL, headers=headers, params=params)
    data = response.json()

    coins = []
    for coin in data["data"]:
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
