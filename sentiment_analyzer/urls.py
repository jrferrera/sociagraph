from django.conf.urls import patterns, url

from sentiment_analyzer import views

urlpatterns = patterns('',
    url(r'^$', views.index, name='index'),
    url(r'^results/$', views.results, name='results'),
    url(r'^upload_corpus/$', views.upload_corpus, name='upload_corpus'),
)