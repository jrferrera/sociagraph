from django.conf.urls import include, url
from django.contrib import admin
from sociagraph import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^key_information_extractor/', include('key_information_extractor.urls', namespace="key_information_extractor")),
    url(r'^sentiment_analyzer/', include('sentiment_analyzer.urls', namespace="sentiment_analyzer")),
    url(r'^admin/', include(admin.site.urls)),
]
