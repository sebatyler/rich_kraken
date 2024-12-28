from django.contrib import admin

from . import models


@admin.register(models.CryptoListing)
class CryptoListingAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
        "symbol",
        "data_at",
        "rank",
        "volume_24h",
        "change_7d",
        "change_24h",
        "change_1h",
        "market_cap",
        "circulating_supply",
        "total_supply",
        "max_supply",
        "price",
        "created",
    )
    list_filter = ("data_at",)
    search_fields = ("=id", "name", "symbol")
    readonly_fields = ("created",)
