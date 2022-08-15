from os import environ
from django.shortcuts import render
from django.views.generic import TemplateView
import krakenex
from pykrakenapi import KrakenAPI

class IndexView(TemplateView):
    template_name = "index.html"

    def get_context_data(self, **kwargs):
        data = super().get_context_data(**kwargs)

        key = environ['KRAKEN_API_KEY']
        secret = environ['KRAKEN_PRIVATE_KEY']
        api = krakenex.API(key, secret)
        k = KrakenAPI(api)
        balance = k.get_account_balance()
        balance = {k: dict(amount=v) for k, v in sorted(balance.to_dict()['vol'].items())}

        print(balance)

        pair = 'XXBTZEUR'
        ticker = k.get_ticker_information(pair).to_dict()
        # average of ask, bid
        price = (float(ticker['a'][pair][0]) + float(ticker['b'][pair][0])) / 2
        balance['XXBT']['price'] = price

        for v in balance.values():
            if 'price' in v:
                v['my_price'] = f"{v['price'] * v['amount']:,.2f} Euros"

        data['balance'] = balance

        # buy Bitcoin
        pair = 'XXBTZEUR'
        btc_price = balance['XXBT']['price']
        # 10 euros
        amount = 10 / btc_price
        print(btc_price, amount)

        r = k.add_standard_order(pair=pair, type="buy", ordertype="market", volume=amount, validate=True)
        print(r)

        return data
