from functools import cached_property
from os import environ
import krakenex
from pykrakenapi import KrakenAPI

class Kraken:
    @cached_property
    def api(self):
        key = environ['KRAKEN_API_KEY']
        secret = environ['KRAKEN_PRIVATE_KEY']
        api = krakenex.API(key, secret)
        return KrakenAPI(api)

    def get_account_balance(self):
        balance = self.api.get_account_balance()
        return {k: dict(amount=v) for k, v in sorted(balance.to_dict()['vol'].items())}

    def get_ticker(self, pair):
        return self.api.get_ticker_information(pair).to_dict()
