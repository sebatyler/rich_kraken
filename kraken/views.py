from django.views.generic import TemplateView

from .models import Kraken

class IndexView(TemplateView):
    template_name = "index.html"

    def get_context_data(self, **kwargs):
        data = super().get_context_data(**kwargs)

        kraken = Kraken()
        balance = kraken.get_account_balance()

        pair = 'XXBTZEUR'
        ticker = kraken.get_ticker(pair)

        # average of ask, bid
        price = (float(ticker['a'][pair][0]) + float(ticker['b'][pair][0])) / 2
        balance['XXBT']['price'] = price

        for v in balance.values():
            if 'price' in v:
                v['my_price'] = f"{v['price'] * v['amount']:,.2f} Euros"

        data['balance'] = balance

        return data
