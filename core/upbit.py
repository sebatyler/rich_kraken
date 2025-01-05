import hashlib
import os
import uuid
from collections import defaultdict
from decimal import Decimal
from urllib.parse import unquote
from urllib.parse import urlencode

import jwt
import requests

from core.utils import dict_omit
from core.utils import dict_pick

access_key = os.getenv("UPBIT_ACCESS_KEY")
secret_key = os.getenv("UPBIT_SECRET_KEY")
origin = "https://api.upbit.com"


def _get_headers(params: dict = None) -> dict:
    """공통 헤더를 생성하는 헬퍼 함수"""
    payload = {
        "access_key": access_key,
        "nonce": str(uuid.uuid4()),
    }

    if params:
        query_string = unquote(urlencode(params, doseq=True)).encode("utf-8")
        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload.update(
            query_hash=query_hash,
            query_hash_alg="SHA512",
        )

    jwt_token = jwt.encode(payload, secret_key)
    return {
        "Authorization": f"Bearer {jwt_token}",
    }


def _request(endpoint: str, method: str = "GET", params: dict = None) -> dict:
    """API 요청을 처리하는 공통 함수"""
    headers = _get_headers(params)
    response = requests.request(
        method=method,
        url=f"{origin}{endpoint}",
        headers=headers,
        params=params,
    )
    return response.json()


def get_balances() -> dict:
    """계좌 잔고 조회"""
    return _request("/v1/accounts")


def get_closed_orders() -> dict:
    """완료된 주문 조회"""
    params = {"state": "done"}
    return _request("/v1/orders/closed", params=params)


def get_withdraws(page: int = 1) -> dict:
    """출금 내역 조회"""
    params = {"state": "DONE", "page": page}
    return _request("/v1/withdraws", params=params)


def get_deposits() -> dict:
    """입금 내역 조회"""
    params = {"currency": "KRW"}
    return _request("/v1/deposits", params=params)


def get_staking_coins():
    """스테이킹 코인 조회"""
    stakings = defaultdict(Decimal)
    page = 1

    while True:
        withdraws = get_withdraws(page=page)
        if not withdraws:
            break

        for withdraw in withdraws:
            if withdraw["transaction_type"] == "internal" and withdraw["txid"].startswith("staking"):
                stakings[withdraw["currency"]] += Decimal(withdraw["amount"])

        page += 1

    return {k: float(v) for k, v in stakings.items()}


def get_available_balances() -> dict:
    """사용 가능한 잔고 조회"""
    balances = {}
    for balance in get_balances():
        symbol = balance["currency"]
        if symbol == "KRW" or float(balance["avg_buy_price"]):
            balances[symbol] = {
                "quantity": balance["balance"],
                "avg_buy_price": balance["avg_buy_price"],
            }

    for symbol, amount in get_staking_coins().items():
        balances[symbol] = {"quantity": amount, "is_staking": True}

    return balances


def get_ticker(ticker):
    data = _request("/v1/ticker", params={"markets": f"KRW-{ticker}"})
    return [dict_omit(row, "market") for row in data]
