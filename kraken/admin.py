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
