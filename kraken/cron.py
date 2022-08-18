from os import environ
from kraken.models import Kraken
import telegram

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
    order = r['descr']['order']

    # current balance and value after order
    balance = kraken.get_account_balance()
    btc_amount = balance['XXBT']['amount']
    btc_value = btc_amount * btc_price

    bot = telegram.Bot(environ['TELEGRAM_BOT_TOKEN'])
    bot.sendMessage(chat_id=environ['TELEGRAM_BOT_CHANNEL_ID'], text=f"Order: {order}\nBTC: {btc_amount} {btc_value:,.2f} Euros")