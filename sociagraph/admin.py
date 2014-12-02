from django.contrib import admin
from sociagraph.models import Classified_Corpus
from sociagraph.models import Sentiment_Corpus

admin.site.register(Classified_Corpus)
admin.site.register(Sentiment_Corpus)