import os

import telegram

# https://t.me/RichSebaBot
bot = telegram.Bot(os.getenv("TELEGRAM_BOT_TOKEN"))


def send_message(text, second=False, is_markdown=False, **kwargs):
    """텔레그램으로 메시지를 전송합니다."""
    chat_id = os.getenv("TELEGRAM_BOT_CHANNEL_ID_2ND" if second else "TELEGRAM_BOT_CHANNEL_ID")
    parse_mode = telegram.ParseMode.MARKDOWN_V2 if is_markdown else None
    bot.sendMessage(chat_id=chat_id, text=text, parse_mode=parse_mode, **kwargs)
