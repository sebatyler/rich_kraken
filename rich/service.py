import json
import logging
import re
from datetime import timedelta

import pandas as pd
from pydantic import BaseModel
from pydantic import Field

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

from core import coinone
from core import crypto
from core import utils
from core.llm import invoke_llm
from core.telegram import send_message
from trading.models import Trading
from trading.models import TradingConfig

from .models import CryptoListing


class BaseStrippedModel(BaseModel):
    def __init__(self, *args, **kwargs):
        kwargs = {k: v.strip() if isinstance(v, str) else v for k, v in kwargs.items()}
        super().__init__(*args, **kwargs)


class Recommendation(BaseStrippedModel):
    action: str = Field(..., description="The action to take (BUY or SELL)")
    symbol: str = Field(..., description="The symbol of the cryptocurrency")
    amount: int | None = Field(default=None, description="The amount of the cryptocurrency to buy in KRW")
    quantity: float | None = Field(default=None, description="The quantity of the cryptocurrency to sell")
    limit_price: float | None = Field(default=None, description="The limit price for the order")
    reason: str = Field(..., description="The reason for the recommendation")


class MultiCryptoRecommendation(BaseStrippedModel):
    scratchpad: str = Field(..., description="The analysis scratchpad text")
    reasoning: str = Field(..., description="The reasoning text")
    recommendations: list[Recommendation] = Field(..., description="List of recommended cryptocurrency trades")


def collect_crypto_data(symbol: str, start_date: str):
    """특정 암호화폐의 모든 관련 데이터를 수집합니다."""
    ticker = coinone.get_ticker(symbol)
    crypto_price = (float(ticker["best_asks"][0]["price"]) + float(ticker["best_bids"][0]["price"])) / 2

    crypto_data = crypto.get_quotes(symbol)

    input_data = dict(
        ticker,
        circulating_supply=crypto_data["circulating_supply"],
        max_supply=crypto_data["max_supply"],
        total_supply=crypto_data["total_supply"],
        **crypto_data["quote"]["KRW"],
        current_price=crypto_price,
    )

    # 과거 데이터 수집
    historical_data = crypto.get_historical_data(symbol, "KRW", 30)
    df = pd.DataFrame(historical_data)
    df = df.drop(columns=["conversionType", "conversionSymbol"])
    crypto_data_csv = df.to_csv(index=False)

    # 네트워크 데이터 (비트코인만)
    network_stats_csv = ""
    if symbol == "BTC":
        network_stats = crypto.get_network_stats()
        df = pd.DataFrame(network_stats, index=[0])
        network_stats_csv = df.to_csv(index=False)

    # 뉴스 데이터
    crypto_news = crypto.fetch_news(start_date, symbol)
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
    }


def get_multi_recommendation(
    crypto_data_list: list[dict],
    indices_csv: str,
    balances: dict[str, dict],
    markets: dict[str, dict],
    trading_config: TradingConfig,
) -> MultiCryptoRecommendation:
    """LLM을 사용하여 암호화폐 투자 추천을 받습니다."""
    # 각 코인별 데이터를 하나의 문자열로 조합
    data_descriptions = []
    for data in crypto_data_list:
        symbol = data["symbol"]
        description = f"""
=== {symbol} Data ===
User's current balance data in KRW in JSON
```json
{symbol}_balance_json
```
Market data in JSON
```json
{symbol}_market_json
```
Recent trading data in KRW in JSON
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
            rf"^({symbol}_(json_data|crypto_data_csv|network_stats_csv|crypto_news_csv|balance_json|market_json))",
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

    # 각 코인별 데이터를 개별 변수로 전달하기 위한 kwargs 구성
    kwargs = {
        "indices_csv": indices_csv,
    }

    # 각 코인별로 데이터 변수 추가
    for data in crypto_data_list:
        symbol = data["symbol"]
        balance = balances.get(symbol, {})
        market = markets.get(symbol, {})
        # 매도 시 필요한 정보만 추출
        market = {k: v for k, v in market.items() if "qty" in k}
        kwargs.update(
            {
                f"{symbol}_balance_json": json.dumps(balance),
                f"{symbol}_json_data": json.dumps(data["input_data"]),
                f"{symbol}_market_json": json.dumps(market),
                f"{symbol}_crypto_data_csv": data["crypto_data_csv"],
                f"{symbol}_crypto_news_csv": data["crypto_news_csv"],
            }
        )
        if data["network_stats_csv"]:  # BTC인 경우
            kwargs[f"{symbol}_network_stats_csv"] = data["network_stats_csv"]

    krw_balance = int(float(balances["KRW"]["available"] or 0))
    prompt = f"""You are a crypto trading advisor who is aggressive yet risk-aware. You have access to:
 - Real-time data, historical prices, volatility, news, sentiment
 - Current KRW balance: {krw_balance:,} KRW
 - Min trade: {trading_config.min_trade_amount:,} KRW, step: {trading_config.step_amount:,} KRW

Key Rules (Condensed):
1) Recommendations:
   - {trading_config.min_coins} ~ {trading_config.max_coins} trades (BUY/SELL), or 0 if no good opportunities.
2) BUY constraints:
   - amount ≥ {trading_config.min_trade_amount}, multiple of {trading_config.step_amount}
   - Single BUY ≤ 30% of available KRW, total BUY ≤ 50% of KRW.
3) SELL constraints:
   - quantity must respect exchange increments (qty_unit) and min_qty~max_qty range
   - Consider partial selling if large holdings, to manage risk and slippage
   - limit_price ~ 0.1~0.3% below current for execution
4) Fees & Profit:
   - Fee: 0.02% each trade (0.04% round-trip)
   - Price must move ≥ 0.06% to surpass fees (add ~0.02% safety margin)
5) Risk & Volatility:
   - Avoid risking >2~3% of total portfolio on a single trade
   - High volatility => smaller positions, possibly more diversification
   - Factor in recent news/sentiment for short-term moves
6) Final KRW Ratio:
   - After ALL recommended BUY/SELL are done, aim for 10%~30% of total portfolio in KRW
   - If below 10% or above 30%, explain (e.g., strong bullish/bearish outlook, waiting for better entries)

Response Format (YAML) in Korean (plain text strings for scratchpad and reasoning):
```yaml
scratchpad: |
  [기술적 분석/시장상황 간단 메모 (한국어). 최대 2000자.]
reasoning: |
  [전체 매매 전략 및 이유 (한국어). 최대 2000자.]
recommendations:
  - action: "BUY"    # or "SELL"
    symbol: "BTC"
    amount: 500000   # (int or null) for BUY only
    quantity: null   # (float or null) for SELL only
    limit_price: null  # (int or null) for SELL only
    reason: "예: 강세장 분석, 수수료 고려, 변동성 낮음. 20% 배팅."
```
No extra fields. Keep (scratchpad+reasoning) under 4000 chars. Strictly follow key names.

"""
    if settings.DEBUG:
        with open(f"tmp/{trading_config.user.email.split('@')[0]}.txt", "w") as f:
            f.write(prompt)
            f.write(all_data)
            f.write(json.dumps(kwargs))

    return invoke_llm(MultiCryptoRecommendation, prompt, all_data, **kwargs)


def send_trade_result(
    symbol: str, action: str, amount: float, quantity: float, price: float, balances: dict, chat_id: str, reason: str
):
    """거래 결과를 확인하고 텔레그램 메시지를 전송합니다."""
    crypto_amount = float(balances[symbol]["available"])
    crypto_value = crypto_amount * price
    krw_amount = float(balances["KRW"]["available"])

    price_msg = "{:,.0f}".format(price)
    message_lines = [
        f"{action}: {quantity:,.8f} {symbol} ({amount:,} KRW)",
        f"{crypto_amount:,.5f}{symbol} {crypto_value:,.0f} / {krw_amount:,.0f} KRW",
        f"{symbol} price: {price_msg}KRW",
    ]
    if reason:
        message_lines.append(reason)

    send_message("\n".join(message_lines), chat_id=chat_id)


def process_trade(
    user,
    symbol: str,
    action: str,
    amount: float,
    quantity: float,
    limit_price: float,
    crypto_price: float,
    crypto_balance: dict,
    order_detail: dict,
    chat_id: str,
    reason: str,
):
    """거래를 처리하고 결과를 저장 및 전송합니다."""
    order_data = order_detail["order"]
    Trading.objects.create(
        user=user,
        order_id=order_data["order_id"],
        coin=symbol,
        amount=amount,
        quantity=quantity,
        limit_price=limit_price,
        price=crypto_price,
        type=order_data["type"],
        side=order_data["side"],
        status=order_data["status"],
        fee=order_data["fee"],
        order_detail=order_detail,
    )

    # current balance and value after order
    balances = coinone.get_balances()
    if action == "BUY":
        traded_quantity = float(balances[symbol]["available"]) - float(crypto_balance.get("available") or 0)
        traded_amount = amount
    elif action == "SELL":
        traded_quantity = float(crypto_balance.get("available") or 0) - float(balances[symbol]["available"])
        traded_amount = traded_quantity * crypto_price
    else:
        raise ValueError(f"Invalid action: {action}")

    send_trade_result(
        symbol=symbol,
        action=action,
        amount=traded_amount,
        quantity=traded_quantity,
        price=crypto_price,
        balances=balances,
        chat_id=chat_id,
        reason=reason,
    )

    return balances


def execute_trade(
    user,
    recommendation: Recommendation,
    crypto_data: dict,
    crypto_balance: dict,
    chat_id: str,
) -> dict:
    """거래를 실행하고 결과를 처리합니다."""
    action = recommendation.action
    symbol = recommendation.symbol
    crypto_price = crypto_data["input_data"]["current_price"]

    logging.info(f"{recommendation=}")

    if settings.DEBUG:
        return

    if action == "BUY":
        amount = recommendation.amount
        if not amount:
            raise ValueError("amount is required for buy order")

        order = coinone.buy_ticker(symbol, amount)
    elif action == "SELL":
        quantity = recommendation.quantity
        limit_price = recommendation.limit_price
        if not quantity:
            raise ValueError("quantity is required for sell order")

        order = coinone.sell_ticker(symbol, quantity, limit_price)
    else:
        raise ValueError(f"Invalid action: {action}")

    logging.info(f"{action} order: {order}")

    if not order.order_id:
        raise ValueError(f"Failed to execute {action} order: {order}")

    order_detail = coinone.get_order_detail(order.order_id, symbol)
    logging.info(f"order_detail: {order_detail}")

    return process_trade(
        user=user,
        symbol=symbol,
        action=action,
        amount=recommendation.amount,
        quantity=recommendation.quantity,
        limit_price=recommendation.limit_price,
        crypto_price=crypto_price,
        crypto_balance=crypto_balance,
        order_detail=order_detail,
        chat_id=chat_id,
        reason=recommendation.reason,
    )


def auto_trading():
    """암호화폐 매매 프로세스를 실행합니다."""
    # 오늘 날짜와 한 달 전 날짜 설정
    end_date = timezone.localdate()
    start_date = (end_date - timedelta(days=30)).strftime("%Y-%m-%d")

    # 시장 지표 데이터 가져오기
    indices_data_csv = crypto.get_market_indices(start_date)

    # 전체 종목 정보 가져오기 (qty_unit 정보 포함)
    markets = coinone.get_markets()

    # 활성화된 트레이딩 설정에서 모든 target_coins를 가져와서 중복 제거
    active_configs = TradingConfig.objects.filter(is_active=True)
    target_coins = set()
    for config in active_configs:
        target_coins.update(config.target_coins)

    # 모든 코인의 데이터 수집
    news_start_date = (end_date - timedelta(days=7)).strftime("%Y-%m-%d")
    crypto_data_dict = {}
    for symbol in target_coins:
        try:
            crypto_data = collect_crypto_data(symbol, news_start_date)
            crypto_data_dict[symbol] = crypto_data
        except Exception as e:
            logging.error(f"Failed to collect data for {symbol}: {e}")
            continue

    # 각 활성화된 유저별로 처리
    for config in active_configs:
        config: TradingConfig = config
        chat_id = config.telegram_chat_id

        # initialize coinone
        coinone.init(
            access_key=config.coinone_access_key,
            secret_key=config.coinone_secret_key,
        )

        balances = coinone.get_balances()

        # 해당 유저의 target_coins에 대한 데이터만 필터링
        user_crypto_data = {
            symbol: crypto_data_dict[symbol] for symbol in config.target_coins if symbol in crypto_data_dict
        }

        # LLM에게 추천 받기
        result, exc = [None] * 2
        # 최대 2번 시도
        for _ in range(2):
            try:
                result = get_multi_recommendation(
                    list(user_crypto_data.values()),
                    indices_data_csv,
                    balances,
                    markets,
                    config,
                )
                break
            except Exception as e:
                logging.warning(e)
                exc = e

        if not result and exc:
            logging.exception(f"Error getting multi recommendation for {config.user}: {exc}")
            continue

        # 분석 결과 전송
        send_message(
            f"```\n코인 분석:\n{result.scratchpad}\n\n{result.reasoning}```",
            chat_id=chat_id,
            is_markdown=True,
        )

        # 추천받은 거래 실행
        for recommendation in result.recommendations:
            symbol = recommendation.symbol
            crypto_data = user_crypto_data[symbol]
            crypto_balance = balances[symbol]

            try:
                new_balances = execute_trade(
                    user=config.user,
                    recommendation=recommendation,
                    crypto_data=crypto_data,
                    crypto_balance=crypto_balance,
                    chat_id=chat_id,
                )
                if new_balances:
                    balances = new_balances
            except Exception as e:
                logging.exception(f"Error executing trade for {symbol}: {e}")


def fetch_crypto_listings():
    """CoinMarketCap에서 암호화폐 목록을 가져와 저장합니다."""
    min_market_cap = 1_000
    data = crypto.get_latest_listings(min_market_cap=min_market_cap)

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

    result = CryptoListing.objects.bulk_create(listing)
    logging.info(f"fetch_crypto_listings: {len(result)}")


def select_coins_to_buy():
    """구매할 코인을 선택하고 결과를 알립니다."""
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

        market_cap = utils.format_currency(coin["avg_market_cap"])
        text_list.extend([f"Average Market Cap: {market_cap}", ""])

    if text_list:
        text = "\n".join(text_list)
        text = f"Selected Coins to Buy:\n```\n{text}```"
    else:
        text = "No coins met the criteria for buying\."

    config = TradingConfig.objects.filter(is_active=True, user__is_superuser=True).first()
    send_message(text, chat_id=config.telegram_chat_id, is_markdown=True)
