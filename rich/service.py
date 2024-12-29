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
from trading.models import TradingConfig

from .models import CryptoListing


class BaseStrippedModel(BaseModel):
    def __init__(self, *args, **kwargs):
        kwargs = {k: v.strip() if isinstance(v, str) else v for k, v in kwargs.items()}
        super().__init__(*args, **kwargs)


class MultiCryptoRecommendation(BaseStrippedModel):
    scratchpad: str = Field(..., description="The analysis scratchpad text")
    reasoning: str = Field(..., description="The reasoning text")
    recommendations: list[dict] = Field(..., description="List of recommended cryptocurrencies and amounts")


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
    crypto_data_list: list[dict], indices_csv: str, balances: dict[str, dict], trading_config: TradingConfig
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
            rf"^({symbol}_(json_data|crypto_data_csv|network_stats_csv|crypto_news_csv|balance_json))",
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
        kwargs.update(
            {
                f"{symbol}_balance_json": json.dumps(balance),
                f"{symbol}_json_data": json.dumps(data["input_data"]),
                f"{symbol}_crypto_data_csv": data["crypto_data_csv"],
                f"{symbol}_crypto_news_csv": data["crypto_news_csv"],
            }
        )
        if data["network_stats_csv"]:  # BTC인 경우
            kwargs[f"{symbol}_network_stats_csv"] = data["network_stats_csv"]

    prompt = f"""You are an aggressive cryptocurrency investment advisor focusing on high-potential daily trading opportunities. You have access to real-time trading data, historical prices, market trends, and news for each cryptocurrency.

Investment amount rules:
- Minimum amount: {trading_config.min_amount:,} KRW
- Maximum amount: {trading_config.max_amount:,} KRW
- Step amount: {trading_config.step_amount:,} KRW
- Amount must be a multiple of step amount
- Focus on coins showing strong momentum or potential reversal signals
- Use higher amounts (near max) for strong setups with lower risk
- Use medium amounts for good setups with moderate risk
- Use lower amounts (near min) for higher risk opportunities

Your task:
1. Analyze each cryptocurrency's data considering:
   - Current price and trading volume
   - Price trends over the last 30 days
   - Recent news impact and market sentiment
   - Technical indicators and patterns
   - Risk level and market volatility

2. Recommend 1-2 cryptocurrencies with the highest potential for gains
   - MUST recommend at least 1 cryptocurrency to buy
   - Focus on coins showing strong momentum or potential reversal signals
   - Investment amount should reflect both opportunity and risk

The output MUST strictly follow this YAML format:
```yaml
scratchpad: |
  [Write detailed analysis here in plain text, no markdown or special characters]

reasoning: |
  [Write summary and reasoning here in plain text, no markdown or special characters]

recommendations:
  - symbol: "SYMBOL1"
    amount: NUMBER1
  - symbol: "SYMBOL2"
    amount: NUMBER2
```

Example output:
```yaml
scratchpad: |
  DOGE 분석:
  현재가: 123.45 KRW (전일대비 -5%)
  거래량: 최근 4시간 동안 200% 급증
  모멘텀: RSI 과매도 구간, 반등 가능성
  리스크: 변동성 높음, 단기 상승 여력 있음
  투자의견: 반등 가능성 높으나 리스크 고려하여 중간 수준 투자

  SOL 분석:
  현재가: 456.78 KRW (전일대비 +3%)
  거래량: 꾸준한 상승세, 전주 대비 80% 증가
  모멘텀: 상승 추세 지속 중, MACD 상향 돌파
  리스크: 안정적 상승세, 기술적 지표 양호
  투자의견: 강한 상승 모멘텀과 낮은 리스크로 적극적 투자 권장

reasoning: |
  DOGE: 과매도 상태에서 거래량 급증하며 반등 신호. 변동성 위험 고려하여 중간 수준 투자
  SOL: 안정적인 상승세와 기술적 지표 양호. 상승 모멘텀 강화로 적극적 투자 권장

recommendations:
  - symbol: "DOGE"
    amount: 10000
  - symbol: "SOL"
    amount: 25000
```

Critical format rules:
1. scratchpad and reasoning must be plain text with no markdown or special characters
2. recommendations must be a list of dictionaries with exact keys: 'symbol' and 'amount'
3. symbol must be a string in quotes, amount must be a number without quotes
4. Keep the exact YAML indentation as shown
5. Do not add any extra fields or formatting

Remember:
1. Write analysis and reasoning in Korean
2. Balance risk and reward when determining amounts
3. MUST recommend at least 1 coin to buy
4. Consider both technical and fundamental factors
5. This analysis runs daily - focus on opportunities"""

    return invoke_llm(MultiCryptoRecommendation, prompt, all_data, **kwargs)


def buy_crypto():
    """암호화폐 구매 프로세스를 실행합니다."""
    # 오늘 날짜와 한 달 전 날짜 설정
    end_date = timezone.localdate()
    start_date = (end_date - timedelta(days=30)).strftime("%Y-%m-%d")

    # 시장 지표 데이터 가져오기
    indices_data_csv = crypto.get_market_indices(start_date)

    # 활성화된 트레이딩 설정에서 모든 target_coins를 가져와서 중복 제거
    active_configs = TradingConfig.objects.filter(is_active=True)
    target_coins = set()
    for config in active_configs:
        target_coins.update(config.target_coins)

    # 모든 코인의 데이터 수집
    crypto_data_dict = {}
    for symbol in target_coins:
        try:
            crypto_data = collect_crypto_data(symbol, start_date)
            crypto_data_dict[symbol] = crypto_data
        except Exception as e:
            logging.error(f"Failed to collect data for {symbol}: {e}")
            continue

    # 각 활성화된 유저별로 처리
    for config in active_configs:
        config: TradingConfig = config
        chat_id = config.telegram_chat_id

        try:
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
            for _ in range(3):
                try:
                    result = get_multi_recommendation(
                        list(user_crypto_data.values()),
                        indices_data_csv,
                        balances,
                        config,
                    )
                    break
                except Exception as e:
                    logging.warning(e)
                    exc = e

            if not result and exc:
                raise exc

            # 분석 결과 전송
            send_message(
                f"```\n코인 분석:\n{result.scratchpad}\n\n{result.reasoning}```",
                chat_id=chat_id,
                is_markdown=True,
            )

            # 추천받은 코인 구매
            for recommendation in result.recommendations:
                symbol = recommendation["symbol"]
                amount = recommendation["amount"]

                if amount == 0:
                    continue

                crypto_data = user_crypto_data[symbol]
                crypto_price = crypto_data["input_data"]["current_price"]
                crypto_balance = crypto_data["input_data"]["my_crypto_balance"]

                # buy crypto
                logging.info(f"{symbol} {crypto_price=}, {amount=}")

                if not settings.DEBUG:
                    r = coinone.buy_ticker(symbol, amount)
                    logging.info(f"buy_ticker: {r}")

                # current balance and value after order
                balances = coinone.get_balances()
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

                send_message("\n".join(message_lines), chat_id=chat_id)

        except Exception as e:
            logging.exception(f"Error processing user {config.user}: {e}")


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

    config = TradingConfig.objects.filter(is_active=True, is_superuser=True).first()
    send_message(text, chat_id=config.telegram_chat_id, is_markdown=True)
