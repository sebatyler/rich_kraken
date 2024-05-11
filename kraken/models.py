import os

import krakenex
from model_utils.models import TimeStampedModel
from pykrakenapi import KrakenAPI

from django.db import models
from django.utils.functional import cached_property


class Kraken:
    @cached_property
    def api(self):
        key = os.getenv("KRAKEN_API_KEY")
        secret = os.getenv("KRAKEN_PRIVATE_KEY")
        api = krakenex.API(key, secret)
        return KrakenAPI(api)

    def get_account_balance(self):
        balance = self.api.get_account_balance()
        return {k: dict(amount=v) for k, v in sorted(balance.to_dict()["vol"].items()) if v > 0}

    def get_ticker(self, pair):
        return self.api.get_ticker_information(pair).to_dict()

    def get_trades(self, start, end=None):
        return self.api.get_trades_history(start=start, end=end)


class Trade(TimeStampedModel):
    txid = models.CharField(max_length=50)
    pair = models.CharField(max_length=20)
    trade_at = models.DateTimeField()
    order_type = models.CharField(max_length=20)
    price = models.FloatField()
    cost = models.FloatField()
    volume = models.FloatField()
    fee = models.FloatField()
    margin = models.FloatField()
    misc = models.CharField(max_length=50)
    raw = models.JSONField(default=dict)

    def __str__(self):
        return f"Trade({self.id}): {self.pair}/{self.order_type}/{self.volume} @ {self.price}"

    @property
    def spent(self):
        return self.cost + self.fee


class CryptoListing(TimeStampedModel):
    name = models.CharField(max_length=50)
    symbol = models.CharField(max_length=10)
    data_at = models.DateTimeField()
    rank = models.IntegerField()
    market_cap = models.FloatField()
    circulating_supply = models.FloatField()
    total_supply = models.FloatField()
    max_supply = models.FloatField(blank=True, null=True)
    price = models.FloatField()
    market_cap = models.FloatField()
    change_1h = models.FloatField()
    change_24h = models.FloatField()
    change_7d = models.FloatField()
    volume_24h = models.FloatField()
    raw = models.JSONField(default=dict)

    def __str__(self):
        return f"CryptoListing({self.id}): {self.name}({self.symbol})/{self.price}"
