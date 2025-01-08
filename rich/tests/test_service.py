import pytest

from rich.service import send_trade_result


@pytest.fixture
def balances():
    return {
        "BTC": {
            "available": "1.5",  # 1.5 BTC
            "balance": "1.5",
        },
        "KRW": {
            "available": "10000000",  # 1천만원
            "balance": "10000000",
        },
    }


@pytest.fixture
def mock_send_message(mocker):
    return mocker.patch("rich.service.send_message")


@pytest.mark.django_db
class TestSendTradeResult:
    def test_send_successful_buy_trade_result(self, mock_send_message, trading, balances):
        # Given
        chat_id = "test_chat_id"

        # When
        send_trade_result(trading=trading, balances=balances, chat_id=chat_id)

        # Then
        expected_message = "\n".join(
            [
                "BUY: 0.1 BTC (5,000,000 원)",
                "보유: 1.5 BTC 75,000,000 / 10,000,000 원",
                "BTC 거래 가격: 50,000,000 원",
                trading.reason,
            ]
        )
        mock_send_message.assert_called_once_with(expected_message, chat_id=chat_id)

    def test_send_successful_sell_trade_result(self, mock_send_message, trading_factory, balances):
        # Given
        trading = trading_factory(side="SELL", amount=None, quantity=0.1, limit_price=5_000_000)
        chat_id = "test_chat_id"

        # When
        send_trade_result(trading=trading, balances=balances, chat_id=chat_id)

        # Then
        expected_message = "\n".join(
            [
                "SELL: 0.1 BTC (5,000,000 원)",
                "보유: 1.5 BTC 75,000,000 / 10,000,000 원",
                "BTC 거래 가격: 50,000,000 원",
                trading.reason,
            ]
        )
        mock_send_message.assert_called_once_with(expected_message, chat_id=chat_id)

    def test_send_failed_buy_trade_result(self, mock_send_message, trading, balances):
        # Given
        trading.executed_qty = None
        trading.average_executed_price = None
        chat_id = "test_chat_id"

        # When
        send_trade_result(trading=trading, balances=balances, chat_id=chat_id)

        # Then
        expected_message = "\n".join(
            [
                "BUY: 0 BTC (0 원)",
                trading.reason,
                f"주문 취소됨! 주문하는게 좋다고 판단하면 직접 주문하세요. BUY / 추천 매수금액: {trading.amount:,.0f} 원",
            ]
        )
        mock_send_message.assert_called_once_with(expected_message, chat_id=chat_id)

    def test_send_failed_sell_trade_result(self, mock_send_message, trading_factory, balances):
        # Given
        trading = trading_factory(
            side="SELL",
            amount=None,
            quantity=0.1,
            limit_price=5_000_000,
            executed_qty=None,
            average_executed_price=None,
        )
        chat_id = "test_chat_id"

        # When
        send_trade_result(trading=trading, balances=balances, chat_id=chat_id)

        # Then
        expected_message = "\n".join(
            [
                "SELL: 0 BTC (0 원)",
                trading.reason,
                "주문 취소됨! 주문하는게 좋다고 판단하면 직접 주문하세요. SELL / 추천 매도수량: 0.1 BTC",
            ]
        )
        mock_send_message.assert_called_once_with(expected_message, chat_id=chat_id)
