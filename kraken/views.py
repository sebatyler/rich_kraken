import json
from datetime import timedelta

import pandas as pd

from django.conf import settings
from django.utils import timezone
from django.views.generic import TemplateView

from .models import Kraken
from .models import Trade


class IndexView(TemplateView):
    template_name = "index.html"

    def get_context_data(self, **kwargs):
        data = super().get_context_data(**kwargs)

        kraken = Kraken()
        balance = kraken.get_account_balance()

        pair = "XXBTZEUR"
        ticker = kraken.get_ticker(pair)

        # average of ask, bid
        price = (float(ticker["a"][pair][0]) + float(ticker["b"][pair][0])) / 2
        balance["XXBT"]["price"] = price

        for v in balance.values():
            if "price" in v:
                v["my_price"] = f"{v['price'] * v['amount']:,.2f} Euros"

        data["chart_data"] = []

        # trade history
        end = timezone.now()
        days = int(self.request.GET.get("days", 30))

        trades = Trade.objects.filter(pair=pair, trade_at__gte=end - timedelta(days=days)).order_by("-trade_at")
        df = pd.DataFrame(list(trades.values()))
        df["spent"] = df["cost"] + df["fee"]
        first_row = df.iloc[0]
        # TODO: consider deposit euro
        euro_balance = balance["ZEUR"]["amount"] - first_row["spent"]
        btc_balance = balance["XXBT"]["amount"] + first_row["volume"]

        df["euro"] = df["spent"].cumsum() + euro_balance
        df["btc"] = btc_balance - df["volume"].cumsum()
        df["btc_euro"] = df["btc"] * df["price"]
        df["total_euro"] = df["btc_euro"] + df["euro"]
        df["date"] = df["trade_at"].dt.tz_convert(settings.TIME_ZONE).dt.strftime("%m-%d")

        reversed_df = (
            df[["date", "volume", "spent", "euro", "btc", "price", "btc_euro", "total_euro"]]
            .iloc[::-1]
            .reset_index(drop=True)
        )
        json_data = reversed_df.to_json(orient="records")
        data["chart_data"] = json.dumps(json_data)

        recent_row = df.iloc[0]
        old_row = df.iloc[-1]
        bought_btc = recent_row["btc"] - old_row["btc"]
        diff_total_euro = recent_row["total_euro"] - old_row["total_euro"]
        profit = diff_total_euro / old_row["total_euro"] * 100
        balance["BTC"] = {
            "bought amount": bought_btc,
            "increased BTC in Euro": recent_row["btc_euro"] - old_row["btc_euro"],
            "spent Euro balance": old_row["euro"] - recent_row["euro"],
            "Total increased value in Euro": diff_total_euro,
            "Profit": f"{profit:.2f} %",
        }
        balance = {k: v for k, v in sorted(balance.items(), key=lambda item: item[0])}
        data["balance"] = balance

        return data
