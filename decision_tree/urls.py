# from django.conf.urls import url
from django.urls import path

from .views import *

urlpatterns = [
    path('dataset', index_dataset, name='index_dataset'),
    path('prediksi', index_prediksi, name='index_prediksi'),
    path('test', api_predict, name='test')
]