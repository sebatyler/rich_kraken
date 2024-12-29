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
    min_coins = models.SmallIntegerField(
        default=1, help_text="Minimum number of coins to recommend (0 means no minimum)"
    )
    max_coins = models.PositiveSmallIntegerField(default=2, help_text="Maximum number of coins to recommend")

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
        if self.min_coins < 0:
            raise ValidationError("Minimum number of coins cannot be negative")
        if self.min_coins > self.max_coins:
            raise ValidationError("Minimum coins must be less than maximum coins")
        if self.max_coins <= 0:
            raise ValidationError("Maximum number of coins must be positive")

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class Trading(TimeStampedModel):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    order_id = models.CharField(max_length=255)
    coin = models.CharField(max_length=255)
    amount = models.DecimalField(max_digits=13, decimal_places=0, help_text="주문 금액 (KRW)")

    type = models.CharField(max_length=20, help_text="주문 유형 (예: MARKET)")
    side = models.CharField(max_length=10, help_text="BUY/SELL")
    status = models.CharField(max_length=50)
    fee = models.DecimalField(max_digits=13, decimal_places=0, help_text="거래 수수료 (KRW)")
    price = models.DecimalField(max_digits=13, decimal_places=0, null=True, blank=True, help_text="주문 가격 (KRW)")
    fee_rate = models.DecimalField(max_digits=5, decimal_places=4, null=True, blank=True, help_text="수수료율 (%)")
    average_executed_price = models.DecimalField(
        max_digits=13, decimal_places=0, null=True, blank=True, help_text="평균 체결 가격 (KRW)"
    )
    average_fee_rate = models.DecimalField(
        max_digits=5, decimal_places=4, null=True, blank=True, help_text="평균 수수료율 (%)"
    )
    limit_price = models.DecimalField(
        max_digits=13, decimal_places=0, null=True, blank=True, help_text="체결 가격 한도 (KRW)"
    )
    original_qty = models.DecimalField(
        max_digits=17, decimal_places=8, null=True, blank=True, help_text="최초 주문 수량 (코인)"
    )
    executed_qty = models.DecimalField(
        max_digits=17, decimal_places=8, null=True, blank=True, help_text="체결된 수량 (코인)"
    )
    canceled_qty = models.DecimalField(
        max_digits=17, decimal_places=8, null=True, blank=True, help_text="취소된 수량 (코인)"
    )
    traded_amount = models.DecimalField(max_digits=13, decimal_places=0, null=True, blank=True, help_text="체결된 총액")
    original_amount = models.DecimalField(max_digits=13, decimal_places=0, null=True, blank=True, help_text="주문 총액")
    canceled_amount = models.DecimalField(
        max_digits=13, decimal_places=0, blank=True, null=True, help_text="주문 취소 총액"
    )

    order_detail = models.JSONField(default=dict)

    def __str__(self):
        return f"{self.user.username}'s {self.side} {self.coin} order ({self.order_id})"

    def save(self, *args, **kwargs):
        is_adding = self._state.adding

        if is_adding and self.order_detail and (data := self.order_detail.get("order")):
            for field in (
                "fee_rate",
                "average_executed_price",
                "average_fee_rate",
                "limit_price",
                "original_qty",
                "executed_qty",
                "canceled_qty",
                "traded_amount",
                "original_amount",
                "canceled_amount",
            ):
                setattr(self, field, data.get(field))

        super().save(*args, **kwargs)
