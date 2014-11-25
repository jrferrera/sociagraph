from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    # Examples:
    # url(r'^$', 'sociagraph.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    url(r'^theme_matcher/', include('theme_matcher.urls', namespace="theme_matcher")),
    url(r'^sentiment_analyzer/', include('sentiment_analyzer.urls', namespace="sentiment_analyzer")),
    url(r'^admin/', include(admin.site.urls)),
]
