# Generated by Django 5.1.4 on 2025-01-02 15:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trading', '0009_alter_historicaltradingconfig_min_coins_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='historicaltradingconfig',
            name='min_trade_amount',
            field=models.PositiveIntegerField(default=5000, help_text='거래당 최소 금액 (원), 수수료 효율성 고려', verbose_name='최소 거래금액'),
        ),
        migrations.AddField(
            model_name='trading',
            name='quantity',
            field=models.DecimalField(blank=True, decimal_places=8, help_text='주문 수량 (코인)', max_digits=17, null=True),
        ),
        migrations.AddField(
            model_name='tradingconfig',
            name='min_trade_amount',
            field=models.PositiveIntegerField(default=5000, help_text='거래당 최소 금액 (원), 수수료 효율성 고려', verbose_name='최소 거래금액'),
        ),
        migrations.AlterField(
            model_name='historicaltradingconfig',
            name='is_active',
            field=models.BooleanField(default=True, help_text='체크하면 자동 매매가 활성화됩니다', verbose_name='자동 매매 활성화'),
        ),
        migrations.AlterField(
            model_name='trading',
            name='amount',
            field=models.DecimalField(blank=True, decimal_places=0, help_text='주문 금액 (KRW)', max_digits=20, null=True),
        ),
        migrations.AlterField(
            model_name='trading',
            name='limit_price',
            field=models.DecimalField(blank=True, decimal_places=0, help_text='주문 제한가 (KRW)', max_digits=20, null=True),
        ),
        migrations.AlterField(
            model_name='tradingconfig',
            name='is_active',
            field=models.BooleanField(default=True, help_text='체크하면 자동 매매가 활성화됩니다', verbose_name='자동 매매 활성화'),
        ),
    ]
