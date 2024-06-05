# from django.conf.urls import url
from django.urls import path, include
from .views import (
    OcrApiView,
)

urlpatterns = [
    path('ocr', OcrApiView.as_view()),
]