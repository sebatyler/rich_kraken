from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.views import View
from django.views.generic import TemplateView

from core import upbit
from trading.forms import TradingConfigForm
from trading.models import TradingConfig


class IndexView(TemplateView):
    template_name = "index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        if self.request.user.is_authenticated:
            trading_config = TradingConfig.objects.filter(user=self.request.user).first()
            if not trading_config:
                trading_config = TradingConfig.objects.create(
                    user=self.request.user,
                    is_active=False,
                )

            form = TradingConfigForm(instance=trading_config)
            context["form"] = form

        return context

    def post(self, request, *args, **kwargs):
        form = TradingConfigForm(request.POST, instance=request.user.tradingconfig)
        if form.is_valid():
            form.save()
            if request.headers.get("HX-Request"):
                html = render_to_string(
                    "components/alert.html",
                    {"message": "트레이딩 설정이 업데이트 되었습니다!"},
                    request,
                )
                return HttpResponse(html)

        context = self.get_context_data(**kwargs)
        context["form"] = form
        return self.render_to_response(context)


class UpbitBalanceView(LoginRequiredMixin, View):
    """업비트 잔고를 JSON으로 반환하는 뷰"""

    def get(self, request, *args, **kwargs):
        # 슈퍼유저 또는 특정 쿼리 파라미터가 있는 경우만 접근 가능
        if not request.user.is_superuser and request.GET.get("from") != "chatgpt_seba":
            return JsonResponse({"error": "Permission denied"}, status=403)

        # 업비트 잔고 조회
        balances = upbit.get_available_balances()

        # 총 자산 가치 계산을 위해 현재가 조회
        total_coin_value = 0
        krw = balances.pop("KRW", {})
        krw_value = float(krw.get("quantity", 0))

        for symbol, balance in balances.items():
            # 현재가 조회
            ticker = upbit.get_ticker(symbol)
            if ticker:
                current_price = ticker[0]["trade_price"]
                quantity = float(balance["quantity"])
                value = quantity * current_price
                balance["current_price"] = current_price
                balance["value"] = value
                total_coin_value += value

        # 응답 데이터 구성
        response_data = {
            "balances": balances,
            "total_value": total_coin_value + krw_value,
            "total_coin_value": total_coin_value,
            "krw_value": krw_value,
        }

        return JsonResponse(response_data)
