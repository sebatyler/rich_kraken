from simple_history.admin import SimpleHistoryAdmin

from django.contrib import admin

from core.admin import ModelAdmin

from .models import TradingConfig


@admin.register(TradingConfig)
class TradingConfigAdmin(SimpleHistoryAdmin, ModelAdmin):
    list_display = (
        "id",
        "user",
        "is_active",
        "created",
        "modified",
    )
    list_filter = ("is_active",)
    list_select_related = ("user",)
    search_fields = ("user__username", "user__email")
    raw_id_fields = ("user",)
    list_display_links = ("id", "user")
