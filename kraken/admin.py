from django.contrib import admin

from . import models


@admin.register(models.Trade)
class TradeAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "txid",
        "pair",
        "trade_at",
        "order_type",
        "price",
        "volume",
        "cost",
        "fee",
        "margin",
        "spent",
        "created",
    )
    list_filter = ("pair", "order_type", "trade_at")
    ordering = ("-trade_at",)
    readonly_fields = ("spent", "created")


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
