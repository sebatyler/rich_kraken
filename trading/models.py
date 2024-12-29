from model_utils.models import TimeStampedModel
from simple_history.models import HistoricalRecords

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models


class TradingConfig(TimeStampedModel):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    coinone_access_key = models.CharField(max_length=255)
    coinone_secret_key = models.CharField(max_length=255)
    telegram_chat_id = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    target_coins = models.JSONField(help_text="List of target coins", default=list)
    min_amount = models.PositiveIntegerField(default=5_000, help_text="Minimum amount in KRW to invest")
    max_amount = models.PositiveIntegerField(default=30_000, help_text="Maximum amount in KRW to invest")
    step_amount = models.PositiveIntegerField(default=5_000, help_text="Step amount in KRW for investment increments")

    history = HistoricalRecords()

    class Meta:
        verbose_name = "Trading Configuration"
        verbose_name_plural = "Trading Configurations"

    def __str__(self):
        return f"{self.user.username}'s Trading Config"

    def clean(self):
        if self.min_amount > self.max_amount:
            raise ValidationError("Minimum amount must be less than maximum amount")
        if self.step_amount > self.max_amount:
            raise ValidationError("Step amount must not be greater than maximum amount")
        if self.min_amount <= 0 or self.max_amount <= 0 or self.step_amount <= 0:
            raise ValidationError("All amounts must be positive")

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)
