from django.http import HttpResponse
from django.template.loader import render_to_string
from django.views.generic import TemplateView

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
