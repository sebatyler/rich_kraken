import os

import krakenex
from django.utils.functional import cached_property
from pykrakenapi import KrakenAPI


class Kraken:
    @cached_property
    def api(self):
        key = os.getenv("KRAKEN_API_KEY")
        secret = os.getenv("KRAKEN_PRIVATE_KEY")
        api = krakenex.API(key, secret)
        return KrakenAPI(api)

    def get_account_balance(self):
        balance = self.api.get_account_balance()
        return {k: dict(amount=v) for k, v in sorted(balance.to_dict()["vol"].items())}

    def get_ticker(self, pair):
        return self.api.get_ticker_information(pair).to_dict()

    def get_trades(self, start, end=None):
        return self.api.get_trades_history(start=start, end=end)
