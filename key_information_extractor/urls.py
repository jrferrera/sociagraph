from django.conf.urls import patterns, url

from key_information_extractor import views

urlpatterns = patterns('',
    url(r'^$', views.index, name='index'),
    url(r'^corpus/$', views.corpus, name='corpus'),
    url(r'^results/$', views.results, name='results'),
    url(r'^add_corpus/$', views.add_corpus, name='add_corpus'),
)