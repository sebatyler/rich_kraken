from model_utils.models import TimeStampedModel

from django.db import models


class CryptoListing(TimeStampedModel):
    name = models.CharField(max_length=50)
    symbol = models.CharField(max_length=10)
    data_at = models.DateTimeField()
    rank = models.IntegerField()
    market_cap = models.FloatField()
    circulating_supply = models.FloatField()
    total_supply = models.FloatField()
    max_supply = models.FloatField(blank=True, null=True)
    price = models.FloatField()
    market_cap = models.FloatField()
    change_1h = models.FloatField()
    change_24h = models.FloatField()
    change_7d = models.FloatField()
    volume_24h = models.FloatField()
    raw = models.JSONField(default=dict)

    def __str__(self):
        return f"CryptoListing({self.id}): {self.name}({self.symbol})/{self.price}"
