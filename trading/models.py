from model_utils.models import TimeStampedModel
from simple_history.models import HistoricalRecords

from django.conf import settings
from django.db import models


class TradingConfig(TimeStampedModel):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    coinone_access_key = models.CharField(max_length=255)
    coinone_secret_key = models.CharField(max_length=255)
    telegram_chat_id = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    target_coins = models.JSONField(help_text="List of target coins", default=list)

    history = HistoricalRecords()

    class Meta:
        verbose_name = "Trading Configuration"
        verbose_name_plural = "Trading Configurations"

    def __str__(self):
        return f"{self.user.username}'s Trading Config"
