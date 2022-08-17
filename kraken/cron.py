from kraken.models import Kraken


def buy_bitcoin():
    kraken = Kraken()

    pair = 'XXBTZEUR'
    ticker = kraken.get_ticker(pair)

    # average of ask, bid
    btc_price = (float(ticker['a'][pair][0]) + float(ticker['b'][pair][0])) / 2

    # buy Bitcoin by 10 euros
    amount = 10 / btc_price
    print(btc_price, amount)

    is_test = False
    r = kraken.api.add_standard_order(pair=pair, type="buy", ordertype="market", volume=amount, validate=is_test)
    print(r)

    # TODO: send how much bitcoin bought in telegram

