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
        data['balance'] = dict(sorted(balance.to_dict()['vol'].items()))

        # k.add_standard_order(pair="ATOMXBT", type="buy", ordertype="limit", volume="420.0", price="0.00042", validate=False)

        return data
