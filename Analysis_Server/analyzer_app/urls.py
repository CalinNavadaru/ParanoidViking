from django.urls import path
from .views import AnalyzeEmailView

urlpatterns = [
    path('analyze-email/', AnalyzeEmailView.as_view(), name='analyze-email'),
]
