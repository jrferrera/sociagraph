from django.db import models
	
class English_Dictionary(models.Model):
	word = models.TextField()
	associated_word = models.TextField()

class Part_of_Speech(models.Model):
	short_hand = models.CharField(max_length = 10)
	long_hand = models.CharField(max_length = 300)

class Classified_Corpus(models.Model):
	text = models.TextField()
	theme = models.TextField()

class Sentiment_Corpus(models.Model):
	text = models.TextField()
	emotion = models.TextField()