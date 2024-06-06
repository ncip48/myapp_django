# from django.conf.urls import url
from django.urls import path, include
from .views import *

urlpatterns = [
    path('ocr', index_ocr, name='index_ocr'),
    path('api/v1/ocr', OcrApiView.as_view()),
]