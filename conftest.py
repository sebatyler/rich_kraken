from decimal import Decimal

import factory
from factory.django import DjangoModelFactory
from faker import Faker
from pytest_factoryboy import register

from accounts.models import User
from trading.models import Trading

fake = Faker()


@register
class UserFactory(DjangoModelFactory):
    class Meta:
        model = User

    username = factory.LazyFunction(lambda: fake.user_name())
    email = factory.LazyAttribute(lambda obj: f"{obj.username}@example.com")


@register
class TradingFactory(DjangoModelFactory):
    class Meta:
        model = Trading

    user = factory.SubFactory(UserFactory)
    order_id = factory.Sequence(lambda n: f"order_{n}")
    coin = "BTC"
    type = "MARKET"
    side = "BUY"
    status = "done"
    amount = Decimal(5_000_000)  # 500만원
    price = Decimal(50_000_000)  # 5천만원
    fee_rate = Decimal(0.0005)  # 0.05%
    executed_qty = Decimal(0.1)  # 0.1 BTC
    average_executed_price = Decimal(50_000_000)  # 5천만원
    reason = factory.LazyAttribute(lambda obj: f"{obj.side} 주문 사유")
    fee = factory.LazyAttribute(
        lambda obj: obj.average_executed_price * obj.executed_qty * obj.fee_rate if obj.executed_qty else 0
    )
    order_detail = factory.LazyAttribute(
        lambda obj: {
            "order": {
                "order_id": obj.order_id,
                "type": obj.type,
                "side": obj.side,
                "status": obj.status,
                "fee": str(obj.fee),
            }
        }
    )
