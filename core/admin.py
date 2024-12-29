from django.contrib import admin
from django.db.models import ForeignKey
from django.db.models import ManyToManyField


class RawIdFieldsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.raw_id_fields:
            self.raw_id_fields = self.setup_raw_id_fields()

    def setup_raw_id_fields(self):
        return tuple(
            field.name for field in self.model._meta.get_fields() if isinstance(field, (ForeignKey, ManyToManyField))
        )


class ModelAdmin(RawIdFieldsMixin, admin.ModelAdmin):
    pass


class TabularInline(RawIdFieldsMixin, admin.TabularInline):
    pass


class StackedInline(RawIdFieldsMixin, admin.StackedInline):
    pass
