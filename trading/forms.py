from django import forms

from .models import TradingConfig

amount_widget = forms.NumberInput(
    attrs={
        "class": "block w-full rounded-md border-2 border-gray-300 bg-gray-50 p-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:bg-white",
        "min": "0",
        "step": "1000",
    }
)

coin_widget_class = "block w-full rounded-md border-2 border-gray-300 bg-gray-50 p-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:bg-white font-mono text-sm"


class TradingConfigForm(forms.ModelForm):
    target_coins = forms.CharField(
        label="거래 코인",
        help_text="거래할 코인 심볼을 쉼표(,)로 구분하여 입력하세요 (예: BTC, ETH)",
        widget=forms.Textarea(
            attrs={
                "rows": 2,
                "class": "block w-full rounded-md border-2 border-gray-300 bg-gray-50 p-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:bg-white font-mono text-sm",
            }
        ),
    )

    class Meta:
        model = TradingConfig
        fields = [
            "target_coins",
            "min_amount",
            "max_amount",
            "step_amount",
            "min_coins",
            "max_coins",
            "min_trade_amount",
            "is_active",
        ]
        widgets = {
            "min_amount": amount_widget,
            "max_amount": amount_widget,
            "step_amount": amount_widget,
            "min_trade_amount": amount_widget,
            "min_coins": forms.NumberInput(
                attrs={
                    "class": coin_widget_class,
                    "min": "0",
                }
            ),
            "max_coins": forms.NumberInput(
                attrs={
                    "class": coin_widget_class,
                    "min": "1",
                }
            ),
            "is_active": forms.CheckboxInput(
                attrs={
                    "class": "h-5 w-5 rounded border-2 border-gray-300 text-indigo-600 focus:ring-indigo-500 shadow-sm"
                }
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.instance and self.instance.target_coins:
            # JSON 리스트를 콤마로 구분된 문자열로 변환
            self.initial["target_coins"] = ", ".join(self.instance.target_coins)

    def clean_target_coins(self):
        value = self.cleaned_data["target_coins"]

        # 콤마로 구분된 문자열을 리스트로 변환
        coins = [coin.strip().upper() for coin in value.split(",") if coin.strip()]

        if not coins:
            raise forms.ValidationError("최소 하나의 코인을 입력해야 합니다")

        return coins
